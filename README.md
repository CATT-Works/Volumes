# Volumes
Volume and Turning Movement Project

# Introduction
This repo is a framework for the Volume and Turning Movement project carried out by [Center of Advanced Transportation Technology](http://www.catt.umd.edu/). 
In the VTM project we are trying to estimate the traffic volumes using GPS data and other sources. The general approach is described in the paper [Estimating historical hourly traffic volumes via machine learning and vehicle probe data: A Maryland case study] (https://www.sciencedirect.com/science/article/pii/S0968090X18314773). This paper is also available on [Arxiv](https://arxiv.org/abs/1711.00721).

The VTM project is both Machine Learning and Data Science project that has many phases. This repository assumes that data are already preprocessed and given in a specific format. Currently it is optimized for INRIX Maryland 2018 Dataset preporcessed by Zachary Vander Laan (built from 04/13/2020)

This is an internal repository. It will not work without data, and the data are not publicly available due to the data agreement. 

# Functionalities

Currently the following functionalities are offered:
- prepare data for Machine Learning (`./volumes/prepare_data.py`)
- train the model using fixed, previously prepared split (`./Example/DeepAndWide.ipynb`)
- train the model with full cross-validation (`./volumes/model_training.py`)
- prepare and save the results (`./volumes/model_training.py`)
- Metrics analysis (`./Errors.ipynb`)

For more information please contact the authors.
