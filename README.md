# Content 

This repository contains the code produced as part of my Bachelor's thesis at the xAI chair at the University of Bamberg. The topic of the thesis was an investigation into the behaviour of DINOv2's CLS token under data covariate shifts. 

# Setup 

## Working Environment

In order to work with the content of this repository, following preperation steps are recommended after the repository was cloned:

1) Create a new conda environment with python installed at a specific version: 
``` 
conda create -n <new_env> python=3.12.9
```

2) Then navigate to the root-folder of the cloned repository. The following command will install all required dependencies, which are defined in the `pyproject.toml` file: 
```
pip install -e . 
``` 

## DINOv2 prerequisites 

For embedding computation `ViT-S/14` of the DINOv2 model family was used. There is no need to explicitly download this model as this is done as part of the code. 

One of the notebooks uses a pretrained head for `ViT-S/14`, which must be downloaded explicitly beforehand. 
The pretrained head can be downloaded from here: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth
The downloaded head should be placed under: `/resources/pretrained_heads/` 

## Datasets 

The analyses made are based on the embeddings of four different datasets: `ImageNet-1k`, `ImageNet-R`, `ImageNet-V2`, and `ImageNet-C`. For `ImageNet-1k` it is enough to compute the embeddings of the validation set. For `ImageNet-R`, `ImageNet-V2`, and `ImageNet-C` the embeddings of all samples must be computed in order to conduct the experiments in the same scope as in the thesis. 
For `ImageNet-1k` and `ImageNet-R` the computed embeddings are already provided in the repository under: `/resources/vit_s_embeddings/`. 

There are three different `ImageNet-V2` datasets: Threshold0.7, TopImages, and MatchedFrequency. All three were used in the experiments. 
These three datasets can be downloaded from here: https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main

The `ImageNet-C` benchmark provides 95 validation datasets, each in size of the original ImageNet-1k validation set. 
The 95 ImageNet-C validation sets can be downloaded from here: https://zenodo.org/records/2235448 






