# Plant Disease Classification with ResNet50

This repository contains a Python script for fine-tuning the pre-trained ResNet50 model for plant disease classification, using TensorFlow and Keras.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Credits](#credits)

## Overview
The `pv_resnet50.ipynb` script imports the necessary libraries for the task, including ResNet50 from Keras Applications. After mounting Google Drive to the Colab environment and unzipping the dataset, the script sets several parameters for data preprocessing and creates data generators for training and validation.

Next, the script defines custom metrics functions and modifies the fully connected layers of the pre-trained ResNet50 model to fit the number of classes in the PlantVillage dataset. The model is compiled and trained on the generator data, with callbacks to monitor early stopping and best models.

Finally, the trained model is saved in h5 format, and utility functions are provided for preprocessing input images and returning predicted class labels.

## Dependencies
The following libraries and versions were used for this project:
- Python
- tensorflow 2
- Keras
- numpy
- matplotlib

## Dataset
The PlantVillage dataset was used for this project, which consists of over 54,000 images of healthy and diseased plants in ~40 different classes. The dataset can be downloaded on [Kaggle - PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

## Usage
To run the script, follow these steps:

1. Clone this repository.
```
git clone https://github.com/gateremark/Plant_Disease_ML_Model1.git
```

2. Download the [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) dataset and place it in your Google Drive.

3. Update the `base_path`, `train_data_dir`, and `val_data_dir` variables in the script to point to the correct directories of the PlantVillage dataset.

4. Open the script in Google Colab, and run the cells.

## Results
The trained ResNet50 model achieved an accuracy of ~95% after 20 epochs of training on the PlantVillage dataset. The model can be used to classify new input images of plants into their respective disease categories.

## Credits
- [PlantVillage-Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [Deep Residual Networks (ResNet, ResNet50)](https://viso.ai/deep-learning/resnet-residual-neural-network/)
- [Victor Ndaba](https://github.com/ndaba1)
