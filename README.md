# B-Cosified Stable Diffusion Version 2

This repository contains a B-cosified version of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).
B-cos networks are a variation of Deep Neural Networks which promote interpretability. 


This is a fork of of the [Stable Diffusion 2.1](https://github.com/Stability-AI/stablediffusion) repository.
The code for training was adapted from an open-source [Dreambooth](https://github.com/tomasyany/dreambooth) repository.
The [b-cos library](https://github.com/B-cos/B-cos-v2/) is used whenever possible, but some modules have also been copied and modified.

________________________________
  
## Requirements

The requirements mostly match the requirements of Stable Diffusion 2.1. 
However, there are some additional package requirements that are most likely currently not correctly specified.

### pre-trained models
Model checkpoints for the vanilla, B-cos x0 and B-cos eps can be found on [Huggingface >> B-cosStableDiffusion](https://huggingface.co/nbernold/B-cosStableDiffusion).
They all share the same architectural shapes (e.g. 6 input channels, 1 ResBlock per downsample etc.).
Vanilla just doesn't use any of the B-cos features and contains trainable biases. B-cos x0 and B-cos eps use the B-cos networks and use the x0- and eps-parameterization, respectively.
The codebase was adapted to support v-parameterization, but we have not trained any model for it.

### xformers efficient attention
While the xformers library provides many benefits, it is not (yet) compatible with the B-cos explanation mode.
The adaption should be relatively straightforward, but we did not do that yet. 

### incompatible parts of the code
Some settings are currently not supported despite them not being removed from the code.
The example Jupyter notebook should provide some guidance on how things work.



