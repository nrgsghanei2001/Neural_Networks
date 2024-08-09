# Neural Network Projects

This repository contains various projects related to neural networks, ranging from basic perceptron implementations to more advanced deep learning models like Convolutional Neural Networks (CNN) and VGG16. 
Each project is organized in its own directory with relevant code and data files.

## Project Overview

1. **Perceptron for Linear Classification**
2. **Polynomial Regression**
3. **McCulloch-Pitts Neuron for Deterministic Finite Automata (DFA)**
4. **Backpropagation for Gaussian Quantiles**
5. **Autoencoder for MNIST Classification**
6. **Convolutional Neural Networks (CNN) and VGG16 on MNIST**

### 1. Perceptron for Linear Classification

**Objective:** Implement a simple perceptron model to classify linearly separable data.

**Approach:** 
- The perceptron algorithm is used to train a linear classifier. 
- The model iteratively adjusts weights based on misclassified examples until convergence is achieved or a maximum number of iterations is reached.


### 2. Polynomial Regression

**Objective:** Perform regression on polynomial functions.

**Approach:** 
- A neural network model is trained to perform regression on data that follows a polynomial relationship with different degrees.
- The model learns to map inputs to outputs by minimizing the mean squared error (MSE).


### 3. McCulloch-Pitts Neuron for DFA

**Objective:** Implement a McCulloch-Pitts neural model to simulate a Deterministic Finite Automaton (DFA).

**Approach:** 
- The McCulloch-Pitts neuron model, which operates using binary threshold logic, is used to design and simulate a DFA.
- This model demonstrates how simple neural units can be configured to perform logical operations and recognize patterns.


### 4. Backpropagation for Gaussian Quantiles

**Objective:** Use backpropagation to classify data into Gaussian quantiles.

**Approach:**
- A neural network is trained using the backpropagation algorithm to classify data points into different Gaussian quantiles.
- The model adjusts its weights through gradient descent, minimizing the error between the predicted and actual quantiles.


### 5. Autoencoder for Classification on MNIST

**Objective:** Use an autoencoder model to classify handwritten digits from the MNIST dataset.

**Approach:**
- An autoencoder is trained to compress the MNIST images into a lower-dimensional representation and then reconstruct them.
- The encoded features are then used as input to a classifier to predict the digit label.


### 6. Convolutional Neural Networks (CNN) and VGG16 on MNIST

**Objective:** Implement CNN and VGG16 models to classify MNIST digits.

**Approach:**
- A basic CNN is trained from scratch to classify MNIST digits, leveraging convolutional layers, pooling layers, and fully connected layers.
- The VGG16 model, a pre-trained deep learning model known for its depth and effectiveness, is fine-tuned on the MNIST dataset to achieve higher accuracy.
