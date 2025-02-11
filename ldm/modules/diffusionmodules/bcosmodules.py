import torch
import torch.nn as nn
import numpy as np
import bcos
from bcos.modules.common import DetachableModule
import ldm.modules.diffusionmodules.util as util

from typing import Optional, Tuple, Union

import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# We use a custom implementation of BcosConv2d and BcosLinear as B-cosification does not necessarily require unit weights.
# The code is copied from https://github.com/B-cos/B-cos-v2/blob/main/bcos/modules/

class BcosConv2d(DetachableModule):
    """
    BcosConv2d is a 2D convolution with unit norm weights and a cosine similarity
    activation function. The cosine similarity is calculated between the
    convolutional patch and the weight vector. The output is then scaled by the
    cosine similarity.

    See the paper for more details: https://arxiv.org/abs/2205.10268

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : int | tuple[int, ...]
        Size of the convolving kernel
    stride : int | tuple[int, ...]
        Stride of the convolution. Default: 1
    padding : int | tuple[int, ...]
        Zero-padding added to both sides of the input. Default: 0
    dilation : int | tuple[int, ...]
        Spacing between kernel elements. Default: 1
    groups : int
        Number of blocked connections from input channels to output channels.
        Default: 1
    padding_mode : str
        Padding mode. One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        Default: ``'zeros'``
    device : Optional[torch.device]
        The device of the weights.
    dtype : Optional[torch.dtype]
        The dtype of the weights.
    b : int | float
        The base of the exponential used to scale the cosine similarity.
    max_out : int
        Number of MaxOut units to use. If 1, no MaxOut is used.
    **kwargs : Any
        Ignored.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        # special (note no scale here! See BcosConv2dWithScale below)
        B: Union[int, float] = 2,
        max_out: int = 1,
        unit_norm: bool = False,
        **kwargs,  # bias is always False
    ):
        assert max_out > 0, f"max_out should be greater than 0, was {max_out}"
        super().__init__()

        # save everything
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.bias = False
        self.unit_norm = unit_norm

        self.b = B
        self.max_out = max_out

        # check dilation
        if dilation > 1:
            warnings.warn("dilation > 1 is much slower!")
            self.calc_patch_norms = self._calc_patch_norms_slow

        self.linear = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * max_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.patch_size = int(
            np.prod(self.linear.kernel_size)  # internally converted to a pair
        )

    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass implementation.
        Args:
            in_tensor: Input tensor. Expected shape: (B, C, H, W)

        Returns:
            BcosConv2d output on the input tensor.
        """
        return self.forward_impl(in_tensor)

    def forward_impl(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            in_tensor: Input tensor. Expected shape: (B, C, H, W)

        Returns:
            BcosConv2d output on the input tensor.
        """
        # Simple linear layer
        out = self.linear(in_tensor)

        # MaxOut computation
        if self.max_out > 1:
            M = self.max_out
            O = self.out_channels  # noqa: E741
            out = out.unflatten(dim=1, sizes=(O, M))
            out = out.max(dim=2, keepdim=False).values

        # if B=1, no further calculation necessary
        if self.b == 1:
            return out

        # Calculating the norm of input patches: ||x||
        norm = self.calc_patch_norms(in_tensor)
        wnorm = 1
        if not self.unit_norm:
            wnorm = LA.vector_norm(self.linear.weight, dim=(1, 2, 3), keepdim=True)[:,0]
        
        # Calculate the dynamic scale (|cos|^(B-1))
        # Note that cos = (x·ŵ)/||x||
        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norm = norm.detach()

        if self.b == 2:
            dynamic_scaling = maybe_detached_out.abs() / (norm*wnorm)
        else:
            abs_cos = (maybe_detached_out / (norm*wnorm)).abs() + 1e-6
            dynamic_scaling = abs_cos.pow(self.b - 1)

        # put everything together
        out = dynamic_scaling * out  # |cos|^(B-1) (ŵ·x)
        return out

    def calc_patch_norms(self, in_tensor: Tensor) -> Tensor:
        """
        Calculates the norms of the patches.
        """
        squares = in_tensor**2
        if self.groups == 1:
            # normal conv
            squares = squares.sum(1, keepdim=True)
        else:
            G = self.groups
            C = self.in_channels
            # group channels together and sum reduce over them
            # ie [N,C,H,W] -> [N,G,C//G,H,W] -> [N,G,H,W]
            # note groups MUST come first
            squares = squares.unflatten(1, (G, C // G)).sum(2)

        norms = (
            F.avg_pool2d(
                squares,
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                # divisor_override=1,  # incompatible w/ torch.compile
            )
            * self.patch_size
            + 1e-6  # stabilizing term
        ).sqrt_()

        if self.groups > 1:
            # norms.shape will be [N,G,H,W] (here H,W are spatial dims of output)
            # we need to convert this into [N,O,H,W] so that we can divide by this norm
            # (because we can't easily do broadcasting)
            N, G, H, W = norms.shape
            O = self.out_channels  # noqa: E741
            norms = torch.repeat_interleave(norms, repeats=O // G, dim=1)

        return norms

    def _calc_patch_norms_slow(self, in_tensor: Tensor) -> Tensor:
        # this is much slower but definitely correct
        # use for testing or something difficult to implement
        # like dilation
        ones_kernel = torch.ones_like(self.linear.weight)

        return (
            F.conv2d(
                in_tensor**2,
                ones_kernel,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            + 1e-6
        ).sqrt_()

    def extra_repr(self) -> str:
        # rest in self.linear
        s = "B={b}"

        if self.max_out > 1:
            s += ", max_out={max_out}"

        # final comma as self.linear is shown in next line
        s += ","

        return s.format(**self.__dict__)


class BcosLinear(DetachableModule):
    """
    BcosLinear is a linear transform with unit norm weights and a cosine similarity
    activation function. The cosine similarity is calculated between the input
    vector and the weight vector. The output is then scaled by the cosine
    similarity.

    See the paper for more details: https://arxiv.org/abs/2205.10268

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        This is ignored. BcosLinear does not support bias.
    device : Optional[torch.device]
        The device of the weights.
    dtype : Optional[torch.dtype]
        The dtype of the weights.
    b : int | float
        The base of the exponential used to scale the cosine similarity.
    max_out : int
        The number of output vectors to use. If this is greater than 1, the
        output is calculated as the maximum of `max_out` vectors. This is
        equivalent to using a MaxOut activation function.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        b: Union[int, float] = 2,
        max_out: int = 1,
        unit_norm: bool = False
    ) -> None:
        assert not bias
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = False

        self.b = b
        self.max_out = max_out
        self.unit_norm = unit_norm
        self.linear = nn.Linear(
            in_features,
            out_features * self.max_out,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            in_tensor: Input tensor. Expected shape: (*, H_in)

        Returns:
            B-cos Linear output on the input tensor.
            Shape: (*, H_out)
        """
        # Simple linear layer
        out = self.linear(in_tensor)

        # MaxOut computation
        if self.max_out > 1:
            M = self.max_out
            O = self.out_features  # noqa: E741
            out = out.unflatten(dim=-1, sizes=(O, M))
            out = out.max(dim=-1, keepdim=False).values

        # if B=1, no further calculation necessary
        if self.b == 1:
            return out

        # Calculating the norm of input vectors ||x||
        norm = LA.vector_norm(in_tensor, dim=-1, keepdim=True) + 1e-12
        wnorm = 1
        if not self.unit_norm:
            wnorm = LA.vector_norm(self.linear.weight, dim=-1)

        # Calculate the dynamic scale (|cos|^(B-1))
        # Note that cos = (x·ŵ)/||x||
        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norm = norm.detach()

        if self.b == 2:
            dynamic_scaling = maybe_detached_out.abs() / (norm*wnorm + 1e-8)
        else:
            abs_cos = (maybe_detached_out / (norm*wnorm + 1e-8)).abs() + 1e-6
            dynamic_scaling = abs_cos.pow(self.b - 1)

        # put everything together
        out = dynamic_scaling * out  # |cos|^(B-1) (ŵ·x)
        return out

    def extra_repr(self) -> str:
        # rest in self.linear
        s = "B={b}"

        if self.max_out > 1:
            s += ", max_out={max_out}"

        # final comma as self.linear is shown in next line
        s += ","

        return s.format(**self.__dict__)

class _SiLU(DetachableModule):
    
    def __init__(self, inplace : bool = False):
        super().__init__()
        if inplace:
            print("Warning: B-cos SiLU does not support inplace operations")
    
    def forward(self, x):
        s = torch.sigmoid(x)
        if self.detach:
            s = s.detach()
        return x * s

class GELU(DetachableModule):

    def __init__(self, approximate : str = "none"):
        super().__init__()
        self.approximate = approximate
        if approximate not in ["none", "tanh"]:
            RuntimeError("approximate argument must be either none or tanh.")
    
    def forward(self, input):
        return self.gelu(input)
    
    def gelu(a):
        M_SQRT2 = 1.41421356237309504880
        M_SQRT1_2 = 0.70710678118654752440
        M_2_SQRTPI = 1.12837916709551257390
        b = a
        if self.detach:
            b = b.detach()
        if self.approximate == "tanh":
            kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
            kKappa = 0.044715
            b_cube = b * b * b
            inner = kBeta * (b + kKappa * b_cube)
            return 0.5 * a * (1 + torch.tanh(inner))
        elif self.approximate == "none":
            kAlpha = M_SQRT1_2
            return a * 0.5 * (1 + torch.erf(b * kAlpha))

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    if kwargs.get("use_bcos", False):
        del kwargs["use_bcos"]
        return bcos.modules.BcosLinear(*args, **kwargs) # TODO: Add switch for (un-)normalized Linear layer
    kwargs.pop("use_bcos", None)
    kwargs.pop("b", None)
    kwargs.pop("max_out", None)
    return util.linear(*args, **kwargs)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if kwargs.get("use_bcos", False):
        del kwargs["use_bcos"]
        if dims == 2:
            return bcos.modules.BcosConv2d(*args, **kwargs) # TODO: Add switch for (un-)normalized Conv layer
        raise ValueError(f"unsupported dimensions: {dims}")
    else:
        kwargs.pop("use_bcos", None)
        kwargs.pop("B", None)
        kwargs.pop("max_out", None)
        return util.conv_nd(dims, *args, **kwargs)

def SiLU(use_bcos=False):
    if use_bcos:
        return _SiLU
    else:
        return nn.SiLU

def LayerNorm(dim, *args, **kwargs):
    if kwargs.get("use_bcos", False):
        del kwargs["use_bcos"]
        return bcos.modules.norms.NoBias(bcos.modules.norms.DetachableLayerNorm)(dim, *args, **kwargs)
    kwargs.pop("use_bcos", None)
    return nn.LayerNorm(dim, *args, **kwargs)

class GroupNorm32(bcos.modules.norms.DetachableGroupNorm2d):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(num_channels, use_bcos=False, *args, **kwargs):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if use_bcos:
        if not kwargs.get("affine", True):
            return GroupNorm32(32, num_channels, *args, **kwargs)
        return bcos.modules.norms.NoBias(GroupNorm32)(32, num_channels, *args, **kwargs) 
    else:
        return util.GroupNorm32(32, num_channels, *args, **kwargs)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    if isinstance(module, bcos.modules.BcosConv2d) or isinstance(module, bcos.modules.BcosLinear) or isinstance(module, BcosConv2d) or isinstance(module, BcosLinear):
        for p in module.parameters():
            p.detach().normal_(0.0, 1e-8) # we cannot set them to 0
        return module
    for p in module.parameters():
        p.detach().zero_()
    return module