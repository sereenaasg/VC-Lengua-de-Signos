# Sign Language Recognitoin
## Project Description
This project focuses on developing an algorithm for recognizing sign language gestures using computer vision techniques. The goal is to facilitate communication between individuals who can speak orally and those who can only communicate through sign language.
The project follows an incremental development approach, with four different processing stages applied to a dataset of three signs (negatiu, positiu, tijeras), the more complex method was also applied to a dataset containing all the letters of the american sign language alphabet. The initial stage involves region detection, followed by the implementation of various computer vision techniques and the utilization of machine learning algorithms such as k-Nearest Neighbors (k-NN) and Support Vector Machines (SVM). In subsequent stages, more complex deep learning models are employed, aiming to achieve higher accuracy and efficiency in sign language recognition.
This repository contains the source code and datasets used to implement all the methods

## Structure
```
├── README.md
├── data
    ├── augmented_data_all_v2
    ├── augmented_data_test
    ├── augmented_data_train
    ├── frames
    ├── mans
    ├── mans_test
    └── videos
├── notebooks
    ├── neural_network_tf.ipynb
    ├── neural_network_tf_MNIST.ipynb
    ├── notebook.ipynb
    └── real_time_detection
        └── modelMNIST.h5
├── real_time_detection
    ├── camerahands.py               
    ├── knn.sav
    ├── link_dataset.txt
    ├── model.h5
    ├── model_github.h5
    ├── smnist.h5
    └── svm.sav
└── src
    ├── data_augmentation.py
    ├── frames_from_video.py
    ├── hand_detection.py
    └── hog_model.py
```
## Table of Contents
 + [Requirements](#Requirements)
 + [Amazing Contributions](#Amazing-Contributions)
 + [Authors](#Authors)
## Requirements
+ tensorflow
+ mediapipe
+ opencv
+ numpy
+ sklearn
+ skimage
## Amazing Contributions
## Authors
 + Javier Esmoris Cerezuela
 + Oriol Marión Escudé
 + Serena Sánchez García
