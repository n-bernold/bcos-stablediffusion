# B-Cosified Stable Diffusion Version 2

This repository contains a B-cosified version of [Stable Diffusion](https://github.com/Stability-AI/stablediffusion).
B-cos networks are a variation of Deep Neural Networks which promote interpretability. 


This is a fork of of the [Stable Diffusion 2.1](https://github.com/Stability-AI/stablediffusion) repository.
The code for training was adapted from an open-source [Dreambooth](https://github.com/tomasyany/dreambooth) repository.
The [b-cos library](https://github.com/B-cos/B-cos-v2/) is used whenever possible but modules have also been copied and modified.

________________________________
  
## Requirements

The requirements mostly match the requirements of Stable Diffusion 2.1. 
However, the minor additions have and changes have not been properly added to the codebase yet. 


#### xformers efficient attention
While the xformers library provides many benefits, it is not (yet) compatible with the B-cos explanation mode.
The adaption should be relatively straight forward but we did not do that yet. 




