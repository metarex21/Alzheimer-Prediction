# AI-based Alzheimer's Disease Detection using Convolutional Neural Networks

This repository contains code and resources related to the development of machine learning models for the detection of Alzheimer's disease using Convolutional Neural Networks (CNNs).

## Requirements
- Python 3.x 
- Tensorflow   ```pip install tensorflow```
- Keras        ```pip install keras```
- Seaborn      ```pip install seaborn```
- Scikit-learn ```pip install scikit-learn```
- Scikit-image ```pip install scikit-image```
- Pydot        ```pip install pydot```
- Graphviz     [here](https://graphviz.gitlab.io/download/)

For GPU Training:
- CUDA Toolkit [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)





## Dataset [here](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset)
### About Dataset
Alzheimer MRI Preprocessed Dataset (128 x 128)

- The Data is collected from several websites/hospitals/public repositories.
- The Dataset is consists of Preprocessed MRI (Magnetic Resonance Imaging) Images.
- All the images are resized into 128 x 128 pixels.
-The Dataset has four classes of images.
- The Dataset is consists of total 6400 MRI images.
  1. Class - 1: Mild Demented (896 images)
  2. Class - 2: Moderate Demented (64 images)
  3. Class - 3: Non Demented (3200 images)
  4. Class - 4: Very Mild Demented (2240 images)
### Motive
The main motive behind sharing this dataset is to design/develop an accurate framework or architecture for the classification of Alzheimers Disease.

### References

- https://adni.loni.usc.edu/
- https://www.alzheimers.net/
- https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
- https://ieeexplore.ieee.org/document/9521165
- https://catalog.data.gov/dataset/alzheimers-disease-and-healthy-aging-data
- https://www.nature.com/articles/s41598-020-79243-9
- https://cordis.europa.eu/article/id/429468-the-final-epad-dataset-is-now-available-on-the-alzheimer-s-disease-workbench


## Code
The code in this repository includes data preprocessing, image augmentation, model creation and evaluation using CNNs, and transfer learning techniques. The CNN models are implemented using the Keras API in Tensorflow. Transfer learning is performed using pre-trained CNN models such as VGG and ResNet.

## Results
The results of the experiments conducted using CNN models are presented in the form of evaluation metrics such as accuracy, precision, recall, and F1 score. The performance of the models is compared to existing state-of-the-art methods for Alzheimer's detection.

## Conclusion
This repository provides a comprehensive framework for the development and evaluation of machine learning models for Alzheimer's disease detection using CNNs. The code and resources in this repository can be used as a starting point for further research and development in this area.
