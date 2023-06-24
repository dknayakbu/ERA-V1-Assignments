# ERA-V1-Assignments for Session 8
This repository contains all the ERA V1 session 8 Assignments.

In this assignment for session 8, we are building a model to classify the MNIST dataset.

# Code Structure
## Aim of Assignment
Aim of this assignment is to apply 3 different normalisation techniques avaialable able and note their effect on the model performance.

The 3 different normalisation techniques are:
  - Batch Normalisation
  - Group Normalisation
  - Layer Normalisation

## The Model Architecture
The model architecture for this experiment will remain same throughout with only change happening in the normalisation part.

We add Convolution in the form of blocks. Each block will consist of the below:
  - Convoltion
  - Normalisation (To be experimented with)
  - Activation function (ReLU)

The model consists of below block layers:
  - Convolution block 1
    - Parameters: in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0
  - Convolution block 2
    - Parameters: in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0
  - Convolution block 3 with Skip connection
    - Parameters: in_channels=16, out_channels=16, kernel_size=(1, 1), padding=0
    - Here we apply skip connection to the convoltion layer
    - It is done by using "x = x + Conv3"
    - To apply skip coonection, we need to keep the in and out channel size same
    - Also the kernel size should be (1, 1)
  - Pooling 1
  - Convolution block 4
    - Parameters: in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0
  - Convolution block 5
    - Parameters: in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0
  - Convolution block 6 with Skip Connection
    - Parameters: in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0
    - Skip connection is added by using "x = x + Conv3"
    - To apply skip coonection, we need to keep the in and out channel size same
    - Also the kernel size should be (1, 1)
  - Pooling 2
  - Convolution block 7
    - Parameters: in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0
  - Convolution block 8
    - Parameters: in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0
  - Convolution block 9
    - Parameters: in_channels=32, out_channels=10, kernel_size=(3, 3), padding=0
  - Global Average Pooling
  - Convolution block 10 with Skip Connection
    - Parameters: in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0
    - Skip connection is added by using "x = x + Conv3"
    - To apply skip coonection, we need to keep the in and out channel size same
    - Also the kernel size should be (1, 1)

# About Normalisation
## What is Normalisation and what is it used for?
Normalization is a techniques used in deep learning models to standardize the activations of neural network layers. The goal of normalization is to transform features to be on a similar scale.
## Why to use normalisation?
Normalisation is used to improve performance and training statbility of the model. It helps in addressing the internal covariate shift problem, which refers to the change in the distribution of network activations as the model parameters are updated during training.
## Different Normalisations available
The different normalisation techniques available for CNN models are:
  - Batch Normalisation
  - Group Normalisation
  - Layer Normalisation
### Batch Normalisation
  - Batch Normalization operates on a mini-batch of samples. It normalizes the activations within a batch along the batch dimension.
  - The mean and standard deviation are calculated for each feature/channel independently across the batch.
  - It introduces additional learnable parameters (scale and shift) to allow the model to learn the optimal scaling and shifting of the normalized activations.
  - Batch Normalization helps address the internal covariate shift problem, reduces the sensitivity to initialization, and can accelerate training by providing more stable gradients.
  - It is commonly used in convolutional neural networks (CNNs) and fully connected layers.
### Group Normalisation
  - Group Normalization divides the channels into groups and computes normalization statistics within each group.
  - Instead of using the batch dimension, it operates along the channel dimension.
  - The mean and standard deviation are calculated independently for each group and each sample in the mini-batch.
  - GN is useful when the batch size is small or when the activations of different channels within a layer have different statistics.
  - It is beneficial in scenarios where the batch dimension may not capture sufficient statistics for normalization, such as in transfer learning or when training with small batch sizes.
### Layer Normalisation
  - Layer Normalization normalizes the activations within each sample along the feature dimension (layer-wise).
  - It computes the mean and standard deviation for each sample independently, using all the features within that sample.
  - LN is commonly applied to recurrent neural networks (RNNs) and transformer models where the activations vary over time steps or positions but remain constant across the batch dimension.
  - Layer Normalization helps with reducing the impact of the variance of activations across different samples.

# Train and Test Accuracy Comparision
In our experiment we only change the norlisation technique while keeping all the other parameters constant to see the effects.

Here we can see that only Group Normalisation did not have much overfitting while Batch and Layer Normalisation had a very high difference between train and test accuracies indicating overfitting.

## Using Batch Normalisation
Train Accuracy = 80.03%

Test Accuracy = 69.77%

## Using Group Normalisation
Train Accuracy = 72.64%

Test Accuracy = 70.00%

## Using Layer Normalisation
Train Accuracy = 71.75%

Test Accuracy = 69.29%

# Train and Test Accuracy Variation
Here we visualize the how the training & testing accuracy & loss vary with respect to different normalisation functions.
We can observe from the below images that while all the normalisation techniques result in almost similar test accuracy and loss scores, the Batch normalisation gets slightly better training as well as test accuracy scores.

## Using Batch Normalisation
![Alt text](image.png)

## Using Group Normalisation
![Alt text](image-2.png)

## Using Layer Normalisation
![Alt text](image-4.png)


# Misclassified Images samples
Here we visualize the how the misclassified images vary with respect to different normalisation functions.
We can observe here that for Batch Normalisation, the misclassified images are relatively tough to distinguish and the predicted class looks similar to actual class. 

Example: Actual Class-Aeroplane vs Predicted Class-Bird

In similar context for the Layer normalisation, the actual vs predicted classes for misclassified images don't look similar and should have been classified correctly.

## Using Batch Normalisation
![Alt text](image-1.png)

## Using Group Normalisation
![Alt text](image-3.png)

## Using Layer Normalisation
![Alt text](image-5.png)

