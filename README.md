# SCA

## Content

- [Introduction](#introduction)
- [Package requirement](#package-requirement)
- [Installation environment](#Installation-environment)
- [Model training and testing and ISI skin status assessment](#Model-training-and-testing-and-ISI-skin-status-assessment)
- [Data availability](#data-availability)
- [Contact](#Contact)

## Introduction

SCA (Skin Condition Assessment) is a deep learning skin condition classification tool that can detect human skin conditions by integrating genotype data (microbiome characteristics), phenotypic data (host variables), and image data (facial features).

- "data" folder : contains multimodal data
  - microbiome.csv : microbiome features
  - host.csv : host variables
  - group.csv : skin condition
  - images.csv :  facial features
- module : model Code
  - func.py : contains multimodal fusion and feature selection algorithms
- result : the results of the model are here
- main.py : run multimodal algorithms with one click
- resnet.py : resnet18 extracts facial features

## Package requirement

```
torch
pandas
numpy
scikit-learn
seaborn
matplotlib
torchvision
torchaudio
openpyxl
xgboost
```

## Installation environment

```
conda create --name SCA python=3.8
conda activate SCA
pip install -r requirements.txt
```

## Model training and testing and ISI skin status assessment

Run multimodal algorithms with one click.

```
python main.py
```

## Data availability

The sequence data in this study have been submitted to the Sequence Read Archive (https://www.ncbi.nlm.nih.gov/sra) and can be accessed through the BioProject numbers PRJNA750340. For the original host information table, see Supplementary_Material.xlsx

## Contact

All problems please contact Meta-Spec development team: **Xiaoquan Su**  Email: [suxq@qdu.edu.cn](mailto:suxq@qdu.edu.cn)
