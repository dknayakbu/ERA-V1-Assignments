# ERA-V1-Assignments for Session 16
This repository contains all the ERA V1 session 16 Assignments.

In this assignment, we are going to spped up the training process by Training the Transformers Efficiently.

The task at hand is to reach a loss of less than 1.8 on the Englist-French translation dataset.

We will be implementing the below:

## OneCycle Policy
The One Cycle Policy is a learning rate schedule that varies the learning rate during training to achieve faster convergence and better final performance. This policy involves gradually increasing and then decreasing the learning rate during training.


## Dynamic Padding
Dynamic padding is a technique that optimizes the padding of input sequences during training. Instead of padding all sequences to the maximum length in a batch, dynamic padding pads each batch to the length of the longest sequence in that batch. This reduces the amount of unnecessary computation and speeds up training.


## Automatic Mixed Precision
Automatic Mixed Precision (AMP) is a method that combines 16-bit and 32-bit floating-point arithmetic to accelerate training while maintaining model accuracy. This technique takes advantage of hardware acceleration for faster training without sacrificing model quality.


## Parameter Sharing
Parameter sharing is a technique where you share model parameters across layers or model components. This can lead to a reduction in the number of parameters in the model, making it more memory-efficient and potentially improving generalization.

# Training Results and Log




