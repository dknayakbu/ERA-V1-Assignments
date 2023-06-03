# ERA-V1-Assignments for Session 5
This repository contains all the ERA V1 session 5 Assignments.

In this assignment for session 5, we are building a model to classify the MNIST dataset.

## Usage
### 1. S5.ipynb
This is a jupyter notebook file which acts as the main file where we execute the whole program and run our model.
### 2. model.py
This file contains the main neural network model that we will be using for training.
### 3. utils.py
This file contains a collection of small python functions and classes which make common patterns shorter and easier incresing reusability of code. it contains the below functions:

## File wise description
### 1. S5.ipynb
The code has been divided into multiple blocks for better readability and interpretability.
#### 1.1 Code Block 1
Import all the necessary packages inorder to be able to run the program.
#### 1.2 Code Block 2
Checks if CUDA/GPU is available or not.
#### 1.3 Code Block 3
Here we transform the Train and Test data to make it ready or compliant for our model. This includes resizinf, random rotation, random apply, converting to tensor and normalizing.
#### 1.4 Code Block 4
Download the train and test data and tranform them based on the transformation we described in the previous block.
#### 1.5 Code Block 5
Here we set the batch size and other multiple parameters required for data loader such as batch size, shuffle, number of workers, pin memory.
#### 1.6 Code Block 6
Displays a sample random images from train dataset using train loader inoder to validate whether the transformations have worked properly or not.
#### 1.7  Code Block 7
Import the Neural Network model from the model.py file.
Also we show the model summary using torchsummary which shows the total number of parameters along with trainable and non-trainable parameters.
#### 1.8 Code Block 8
Initiate empty lists for collecting data to plot accuracy and loss graphs.
#### 1.9 Code Block 9
Import the traing and test functions from the utils.py file.
#### 1.10 Code Block 10
Here we define all the parameters required for the actual training such as optimizer, scheduler, criterion and number of epochs.
We iterate over the number of epochs and train the model using the "train" and test using "test" function imported from previos block.
The train and test accuracies and appended to the data collection list from block 8.
#### 1.11 Code Block 11
Here we plot the training and testing loss and accuracy per epoch. This is to veryfy whether our model has converged properly or not.

### 2. model.py
This file contains the main neural network model that we will be using for training.

### 3. utils.py
This file contains a collection of small python functions and classes which make common patterns shorter and easier incresing reusability of code. it contains the below functions:
#### 3.1 train
The train function takes the data from the train loader and iterates over it batch wise. Per batch, the model predicts and then calculates the loss and does backpropagation based on optimizer.
#### 3.2 test
The test function takes the test data from test loader and iterates over it batch wise. Per batch it calculates the loss based on actual and correct prections.
#### 3.3 plot_train_test_loss
Plots the train and test loss and accuracy per epoch.

