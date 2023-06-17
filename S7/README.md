# ERA-V1-Assignments for Session 7
This repository contains all the ERA V1 session 5 Assignments.

In this assignment for session 5, we are building a model to classify the MNIST dataset.

## Usage
### S7_Model_1.ipynb

Target:
    - To get the basic code setup right.
    - This would include fixing the data transforms and data loader that include 
        - converting to tensor
        - normalization
        - augmentation
    - We also fix the class structure for the model, training & testing block.
    - Fixing the above would help us develop and experiment multiple models faster as we only need to change the model structure while everything else remains same.
Results:
    - Parameters: 3.9M
    - Best Train Accuracy: 99.98
    - Best Test Accuracy: 99.36
Analysis:
    - The basic code setup is built and fixed now.
    - The basic model where we just increase the parametrs is too large and needs to be reduced.


### S7_Model_2.ipynb

Target:
    - To get the basic skeleton of the model right.
    - We will be using this skeleton to add/remove layers and finetune the parameters.
    - This would include fixing the below in a block which will be repeated:
        - Input Block
        - Convolution
        - Activation function
        - Pooling
    - Also we need bring down the number of parameters to less than 8k.
    - Therefore, we need to keep the channel size small.
Results:
    - The basic code structure is built and fixed now.
    - Parameters: 5000
    - Best Train Accuracy: 99.28
    - Best Test Accuracy: 99.00
Analysis:
    - The basic model structure is fixed.
    - This new much smaller model is performing good.
    - We can see a little overfitting, we will address that in next model.

### S7_Model_3.ipynb

Target:
    - We add batch normalization to the convolution and evaluate the results.
    - It is added to standardize the ta
Results:
    - The basic code structure is built and fixed now.
    - Parameters: 5,112
    - Best Train Accuracy: 99.67
    - Best Test Accuracy: 98.91
Analysis:
    - The training accuracy has increased.
    - The model is still having overfitting.

### S7_Model_4.ipynb
### S7_Model_5.ipynb
### S7_Model_6.ipynb



