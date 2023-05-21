# Sign Language Recognitoin
## Project Description
This project focuses on developing an algorithm for recognizing sign language gestures using computer vision techniques. The goal is to facilitate communication between individuals who can speak orally and those who can only communicate through sign language.
The project follows an incremental development approach, with four different processing stages applied to a dataset of three signs (negatiu, positiu, tijeras), the more complex method was also applied to a dataset containing all the letters of the american sign language alphabet. The initial stage involves region detection, followed by the implementation of various computer vision techniques and the utilization of machine learning algorithms such as k-Nearest Neighbors (k-NN) and Support Vector Machines (SVM). In subsequent stages, more complex deep learning models are employed, aiming to achieve higher accuracy and efficiency in sign language recognition.
This repository contains the source code and datasets used to implement all the methods

## Structure
```
├── README.md
├── Informe.pdf
├── data
    ├── augmented_bin_test           # Binary images of hands detected by hand_detection.py 
    ├── augmented_bin_train     
    ├── augmented_data_all_v2        # RGB Images of hands taken manually and augmented 
    ├── augmented_data_test         
    ├── augmented_data_train
    ├── augmented_data_test2
    ├── augmented_data_train2
    ├── frames                       # frames from videos output of frames_from_video.py
    ├── mans                         # Hand images
    ├── mans_test
    ├── MNIST                        # MNIST Sign Language dataset
    ├── preprocessed                 
    └── videos
├── notebooks
    ├── neural_network_tf.ipynb
    ├── neural_network_tf_MNIST.ipynb
    ├── notebook.ipynb
├── real_time_detection
    ├── camerahands.py               # Mediapipe hand detection and code to try saved models
    ├── knn.sav                      # knn model trained with HoG descriptors
    ├── model.h5                     # CNN model trained with three simple gestures
    ├── modelMNIST.h5                # CNN model trained with the MNIST dataset
    └── svm.sav                      # svm model trained with HoG descriptors
└── src
    ├── data_augmentation.py         # Used to generate augmented data
    ├── frames_from_video.py         # Used to get some frames from input videos
    ├── hand_detection.py            # Process to detect hands and width/height ratio classifier
    └── hog_model.py                 # Classifier using HoG descriptors
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
## Authors
 + Javier Esmoris Cerezuela
 + Oriol Marión Escudé
 + Serena Sánchez García
