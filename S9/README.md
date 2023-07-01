# ERA-V1-Assignments for Session 9
This repository contains all the ERA V1 session 9 Assignments.

In this assignment for session 9, we are building a model to classify the MNIST dataset.

# Code Structure
## Aim of Assignment
Aim of this assignment is to 
  - has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
  - total RF must be more than 44
  - one of the layers must use Depthwise Separable Convolution
  - one of the layers must use Dilated Convolution
  - use albumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

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
![image](https://github.com/dknayakbu/ERA-V1-Assignments/assets/20933037/df4a0878-54ce-49ed-9997-4ffc180a5774)

## Sample of Misclassified Images
![image](https://github.com/dknayakbu/ERA-V1-Assignments/assets/20933037/dd176ca2-5623-49ea-9da4-0e01adf6e373)



