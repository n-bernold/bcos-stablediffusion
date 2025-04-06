"""SAMPLING ONLY."""

import torch
import numpy as np
try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from contextlib import nullcontext
from pytorch_lightning import seed_everything


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    #@torch.no_grad() # TODO: We cannot have this here for the explanation but in the other cases we'd like to keep it...
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               no_grad=True,
               disable_tqdm=False,
               t_remaining=None,
               backup_seed=None,
               return_eps=False,
               **kwargs
               ):
        with torch.no_grad() if no_grad else nullcontext():
            if conditioning is not None:
                if isinstance(conditioning, dict):
                    ctmp = conditioning[list(conditioning.keys())[0]]
                    while isinstance(ctmp, list): ctmp = ctmp[0]
                    cbs = ctmp.shape[0]
                    if cbs != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

                elif isinstance(conditioning, list):
                    for ctmp in conditioning:
                        if ctmp.shape[0] != batch_size:
                            print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

                else:
                    if conditioning.shape[0] != batch_size:
                        print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

            self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
            # sampling
            C, H, W = shape
            size = (batch_size, C, H, W)
            if not disable_tqdm:
                print(f'Data shape for DDIM sampling is {size}, eta {eta}')

            samples, intermediates = self.ddim_sampling(conditioning, size,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, x0=x0,
                                                        ddim_use_original_steps=False,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        dynamic_threshold=dynamic_threshold,
                                                        ucg_schedule=ucg_schedule,
                                                        no_grad=no_grad,
                                                        disable_tqdm=disable_tqdm,
                                                        t_remaining=t_remaining,
                                                        backup_seed=backup_seed,
                                                        return_eps=return_eps
                                                        )
            return samples, intermediates

    #@torch.no_grad() # We cannot have this here for the explanation but in the other cases we'd like to keep it...
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, no_grad=True, disable_tqdm=False, t_remaining=None, backup_seed=None, return_eps=False):
        with torch.no_grad() if no_grad else nullcontext():
            device = self.model.betas.device
            b = shape[0]
            if x_T is None:
                img = torch.randn(shape, device=device)
                if self.model.encode_noise:
                    img[...,3:,:,:] = -img[...,:3,:,:]
                img = self.model.mean + self.model.stdev*img
            else:
                img = x_T

            if timesteps is None:
                timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
            elif timesteps is not None and not ddim_use_original_steps:
                subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
                timesteps = self.ddim_timesteps[:subset_end]

            intermediates = {'x_inter': [img], 'pred_x0': [img]}
            time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
            total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
            if not disable_tqdm:
                print(f"Running DDIM Sampling with {total_steps} timesteps")

            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=disable_tqdm)
            
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)

                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(x0, ts)  
                    img = img_orig * mask + (1. - mask) * img

                if ucg_schedule is not None:
                    assert len(ucg_schedule) == len(time_range)
                    unconditional_guidance_scale = ucg_schedule[i]

                if backup_seed is not None:
                    t,seed = backup_seed
                    if t == index:
                        seed_everything(seed)

                if t_remaining == (index,0):
                    return (img, cond), (lambda x, y : self.ddim_sampling_t(y, shape,
                        x, ddim_use_original_steps=ddim_use_original_steps,
                        callback=callback, timesteps=index+1, quantize_denoised=quantize_denoised,
                        mask=mask, x0=x0, img_callback=img_callback, log_every_t=log_every_t,
                        temperature=temperature, noise_dropout=noise_dropout, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, dynamic_threshold=dynamic_threshold,
                        ucg_schedule=ucg_schedule, no_grad=no_grad, disable_tqdm=disable_tqdm, backup_seed=backup_seed, return_eps=return_eps))

                if t_remaining == index:
                    return (img, cond), (lambda x, y : self.p_sample_ddim(x, y, ts, index=index, use_original_steps=ddim_use_original_steps,
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    dynamic_threshold=dynamic_threshold,
                                    no_grad=no_grad, return_eps=return_eps))
                
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        dynamic_threshold=dynamic_threshold,
                                        no_grad=no_grad)
                img, pred_x0 = outs
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)

            return img, intermediates

    def ddim_sampling_t(self, cond, shape,
                      x_T, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, no_grad=True, disable_tqdm=False, backup_seed=None, return_eps=False):
        device = self.model.betas.device
        b = shape[0]
        img = x_T

        seed_everything(backup_seed[1])

        timesteps = self.ddim_timesteps[:timesteps]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if not disable_tqdm:
            print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=disable_tqdm)
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]
            
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    dynamic_threshold=dynamic_threshold,
                                    no_grad=no_grad, return_eps=return_eps)
            if return_eps:
                return outs
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, pred_x0
    
    # Magically transform one image into another while preserving the gradients
    def retarget_img(self, x, tgt):
        tgt = tgt.detach()
        sec1 = (x[:, :3] + x[:, 3:])*tgt[:, :3]
        sec2 = (x[:, :3] + x[:, 3:])*tgt[:, 3:]
        return torch.cat((sec1, sec2), dim=1)

    # Sum of first three and last three channels should be 1
    def renormalize_img(self, x):
        norm = (x[:, :3]+ x[:, 3:]).detach()
        return torch.cat((x[:, :3] / norm, x[:, 3:] / norm), dim=1)

    # Sum of first three and last three channels should be 0
    def renormalize_eps(self, x, eps=1e-9):
        return torch.cat(((x[:, :3] - x[:, 3:])/2, (x[:, 3:] - x[:, :3])/2), dim=1)

    #@torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None,
                    dynamic_threshold=None, no_grad=True, return_eps=False):
        with torch.no_grad() if no_grad else nullcontext():
            b, *_, device = *x.shape, x.device

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
                #model_output = self.renormalize_eps(model_output)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(c, dict):
                    assert isinstance(unconditional_conditioning, dict)
                    c_in = dict()
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [torch.cat([
                                unconditional_conditioning[k][i],
                                c[k][i]]) for i in range(len(c[k]))]
                        else:
                            c_in[k] = torch.cat([
                                    unconditional_conditioning[k],
                                    c[k]])
                elif isinstance(c, list):
                    c_in = list()
                    assert isinstance(unconditional_conditioning, list)
                    for i in range(len(c)):
                        c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                
                if return_eps:
                    return model_t, model_uncond
                
                if self.model.parameterization == "x0":
                    model_uncond = (x - a_t.sqrt()*model_uncond - (1-a_t.sqrt()) * self.model.mean)/(self.model.stdev * sqrt_one_minus_at)
                    model_t = (x - a_t.sqrt()*model_t - (1-a_t.sqrt()) * self.model.mean)/(self.model.stdev * sqrt_one_minus_at)

                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)


            # current prediction for x_0
            if self.model.parameterization == "eps" or self.model.parameterization == "x0":
                pred_x0 = (x - (1-a_t.sqrt()) * self.model.mean - sqrt_one_minus_at * self.model.stdev * e_t) / a_t.sqrt() 

            elif self.model.parameterization == "full_eps":
                e_t = (e_t - self.model.mean)/self.model.stdev
                pred_x0 = (x - (1-a_t.sqrt()) * self.model.mean - sqrt_one_minus_at * self.model.stdev * e_t) / a_t.sqrt() 

            else:
                raise NotImplementedError("Not B-cosified")
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * self.model.stdev * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise, self.model.encode_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)

            x_prev = a_prev.sqrt() * pred_x0 + (1-a_prev.sqrt()) * self.model.mean + dir_xt + noise * self.model.stdev

            return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        raise NotImplementedError("Not B-cosified")

        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        raise NotImplementedError("Not B-cosified")
        
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
            if self.model.encode_noise:
                noise[...,3:,:,:] = -noise[...,:3,:,:]
            noise = self.model.mean + self.model.stdev * noise
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):
        raise NotImplementedError("Not B-cosified")

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec