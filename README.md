# SCT_ML_03
Cat vs Dog Image Classification (NumPy CNN from Scratch)
Project Overview
This project implements a Convolutional Neural Network (CNN) from scratch using NumPy only, without TensorFlow, Keras, or any other deep learning framework.
It classifies images into two categories: Cat or Dog.

The purpose of this project is to understand and build the core components of a CNN manually — convolution layers, pooling layers, activation functions, and fully connected layers — while handling image preprocessing and training logic.

Features
CNN implemented entirely in NumPy
Custom image preprocessing and normalization
Forward & backward propagation from scratch
Training loop with loss calculation
Accuracy tracking during training
Model testing on unseen images
Dataset
Dataset Used: Kaggle Cats and Dogs Dataset
Classes: 2 (Cat, Dog)
Image Size: Resized to 64×64 for computational efficiency
Preprocessing:
Converted to RGB arrays
Normalized pixel values between 0 and 1
Shuffled and split into training and testing sets
