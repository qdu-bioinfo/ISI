# ISI

## Content

- [Introduction](#introduction)
- [Package requirement](#package-requirement)
- [Installation environment](#Installation-environment)
- [Model training and testing and ISI skin status assessment](#Model-training-and-testing-and-ISI-skin-status-assessment)
- [Data availability](#data-availability)
- [Contact](#Contact)

## Introduction

The skin microecology plays a vital role in maintaining cutaneous health and is intricately linked to host skin phenotypes. However, there remains a lack of precise and quantitative biomarkers for evaluating skin health conditions, making it challenging to identify individuals at potential risk of microecological imbalance. To enable a more holistic and quantitative assessment of facial skin status, we developed a novel Ideal Skin Index (ISI) that fuses multi-modal data of facial images, microbiome profiles, and host skin phenotype features by a deep learning-based framework. Importantly, ISI identified individuals with outwardly healthy facial appearance but significant underlying microbial dysbiosis conditions that conventional diagnostic approaches often overlook. This work enables the detection of such hidden risks, offering new avenues for individualized facial skin health assessment, precision dermatology, and microbiome-informed aesthetic interventions.

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

## ISI skin status assessment

Run multimodal algorithms with one click.

```
python main.py
```

## Data availability

The sequence data in this study have been submitted to the Sequence Read Archive (https://www.ncbi.nlm.nih.gov/sra) and can be accessed through the BioProject numbers PRJNA750340. For the original host information table, see Supplementary_Material.xlsx

## Contact

All problems please contact ISI development team: **Xiaoquan Su**  Email: [suxq@qdu.edu.cn](mailto:suxq@qdu.edu.cn)
