# Visual_Obfuscation_Identification
Repository for the corresponding paper on visual obfuscation identification

## Authors: Sebastian Raubitzek, Sebastian Schrittwieser

## Overview

This repository contains two comprehensive scripts for machine learning and complexity analysis of matrices derived from image data.
Binaries and their images are classified into different categories based on groupings, these groupings are parameterized individually in each script.
The primary components of this project are:

1. **Complexity Metrics Analysis using SVD**: A script that computes various complexity metrics of matrices based on Singular Value Decomposition (SVD) and stores these metrics along with class information for further analysis.
2. **Predictive Obfuscations using ExtraTreesClassifier**: This script leverages the ExtraTreesClassifier for predicting obfuscations of binaries, incorporating data preprocessing, hyperparameter optimization, model evaluation, and interpretability techniques.

## Repository Structure


├── README.md

├── 01_build_complexity_dataset_from_images.py

├── 02_perform_ML_ExtraTrees.py

├── ML_data

├── image_data

│ └── ... contains data to be unpacked

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.6
- pandas==1.1.3
- numpy==1.19.5
- scikit-learn==0.24.2
- scipy==1.5.3
- matplotlib==3.2.2
- seaborn==0.11.0
- imbalanced-learn==0.8.1
- scikit-optimize==0.9.0
