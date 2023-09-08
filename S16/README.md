# ERA-V1-Assignments for Session 16
This repository contains all the ERA V1 session 16 Assignments.

In this assignment, we are going to spped up the training process by Training the Transformers Efficiently.

The task at hand is to reach a loss of less than 1.8 on the Englist-French translation dataset.

We will be implementing the below:

## OneCycle Policy
The One Cycle Policy is a technique used in deep learning to train complex models faster and with fewer iterations. It follows the Cyclical Learning Rate (CLR) to obtain faster training time with regularization effect but with a slight modification. Specifically, it uses one cycle that is smaller than the total number of iterations/epochs and allows the learning rate to decrease several orders of magnitude less than the initial learning rate for the remaining iterations (i.e. last few iterations)


## Dynamic Padding
Dynamic padding is a technique used in natural language processing to optimize the padding process during batch creation. Instead of padding all the samples to the maximum length, dynamic padding limits the number of added pad tokens to reach the length of the longest sequence of each mini-batch. This technique is more efficient than traditional padding because it reduces the amount of unnecessary padding, which speeds up training. 


## Automatic Mixed Precision
Automatic Mixed Precision (AMP) is a technique used in deep learning to speed up training and reduce memory usage by combining different numerical formats in one computational workload. AMP is supported by popular deep learning frameworks such as TensorFlow, PyTorch, and MXNet. In the context of Transformers, AMP is used to train models faster by training data in half-precision floating point (FP16) compared to single-precision floating point (FP32). AMP is similar to FP16 mixed precision, but it uses both single and half-precision representations. AMP automates mixed precision by using a combination of automatic casting and scaling of gradients


## Parameter Sharing
Parameter sharing is a technique used in Transformers to reduce the number of parameters in the model and improve its efficiency. In general, parameter sharing involves using the same set of parameters for multiple layers of the model. There are several ways to implement parameter sharing in Transformers, including:
    - Sharing parameters for one layer with all layers, as in Universal Transformers.
    - Repeating the entire Transformer layer a given number of times, as in classic (ALBERT-like) sharing.
    - Using sequence, cycle, or cycle (rev) strategies to perform weight sharing on Transformer models

# Training Results and Log




