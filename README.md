# Emotion Recognition with Convolutional Neural Networks

This personal project implements an emotion recognition system using Convolutional Neural Networks (CNNs) as part of my personal study of PyTorch. The goal is to classify images of faces into one of seven emotions while leveraging transfer learning for efficient training.

## Overview

This project utilizes MobileNet V2 to classify images into seven different emotions. The dataset consists of labeled images organized by emotion categories, and the model is trained using the PyTorch framework. The data preprocessing includes image loading, resizing, and normalization. The dataset can be downloaded from kaggle: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

## Features

- **Transfer Learning**: Utilizes MobileNet V2 to reduce computational overhead.
- **Custom Data Loader**: Efficient loading and transformation of images for training.
- **Training and Testing Pipeline**: Complete workflow for training and evaluating the model.

## Technologies Used

- Python 3.10
- PyTorch
- OpenCV
- torchvision
- NumPy
