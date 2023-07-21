# ERA-V1-Assignments for Session 10
This repository contains all the ERA V1 session 10 Assignments.

In this assignment for session 10, we are building a model to classify the CIFAR 10 dataset using our custom model.

# Code Structure
Custom ResNet architecture for CIFAR10:
  - PrepLayer: Conv (3 s1, p1) >> BN >> RELU [64k]


## Model Architecture
We divide the model using blocks.
### Block Structure
Each block consists of 3 Conv layers with options for Dilation or Depthwise Separable Conv in the middle Conv layer. In each conv layer the optput size becomes ouble of input size.
  - Conv1 layer
  - Batch Normalization
  - ReLU
  - Conv2 layer (with options for Dilation or Depthwise Separable Conv)
  - Batch Normalization
  -  ReLU
  - Conv3 layer (with Stride=2)
  - ReLU

Finally, combining the blocks in the following fashion:
  - Block1
  - Dropout
  - Block2
  - Dropout
  - Block3
  - Dropout
  - GAP
  - Fully connected layer

## Training and Testing Loss


## Sample of Misclassified Images
