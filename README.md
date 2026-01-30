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
Due to file-size limitations on GitHub the `ImageNet-1k`-embeddings are stored across the two files. In order to create a file that stores the embeddings of the `ImageNet-1k` validation set in a single dictionary, first navigate to `/scripts`. 
Then apply this command: 
``` 
python prepare_inet_1k_val_embeds.py 
``` 

There are three different `ImageNet-V2` datasets: Threshold0.7, TopImages, and MatchedFrequency. All three were used in the experiments. 
These three datasets can be downloaded from here: https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main. Each of the three should be placed after unpacking as is under the `/datasets` folder. 

The `ImageNet-C` benchmark provides 95 validation datasets, each in size of the original ImageNet-1k validation set. 
The 95 ImageNet-C validation sets are distributed among five different files, which can be downloaded from here: https://zenodo.org/records/2235448. 
When unpacking the downloaded files, the result should be placed under `/datasets/ImageNetC/`. 

## Embedding computation 

All computed embeddings are expected to be stored under `/resources/vit_s_embeddings/`. Only in case of `ImageNet-C` the embeddings are stored under an additional folder `/resources/vit_s_embeddings/imagenet_c`.  

For the computation of the `ImageNet-V2`-embeddings the notebook `compute_embeddings.py` for example can be employed. For the computation of `ImageNet-C`-embeddings it is advisable to use the script `compute_inet_c_embeds.py` as this is solely designed for that task. 

# Experiments 

The notebook `compute_statistics` measures changes in the embedding space induced by covariate shift. In the first part of the notebook, 200 combinations of classes are defined, which form the basis for the analysis of embedding changes. Four different measures are applied to each class combination to quantify the difference between the embeddings of a combination. Finally, the results across all 200 class combinations are summarized and visualized in diagrams. To reproduce the results described in my thesis, `class_a_classes_b_thesis.csv` should be used instead of `class_a_classes_b.csv` for the defining the class-A-to-class-B mappings. 

The notebook `linear_classifier` employs the pretrained head to compute the accuracy for each of the OOD datasets used. The accuracy values are set in relation to the results of the `compute_statistics` notebook.

The notebooks `example_pictures` and `example_diagrams` are both intended to provide a more detailed look at the methodological approach used in `compute_statistics`. The `example_pictures`  notebook can be used to retrieve samples for a specific class combination. The `example_diagrams` notebook applies the four measures to a specific class combination and visualizes the results in the same style as in the `compute_statistics` notebook. 






