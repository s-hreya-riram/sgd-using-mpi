import numpy as np
from scipy.special import expit

def linear(x): return x
def linear_grad(z): return np.ones_like(z)

def relu(x): return np.maximum(0, x)
def relu_grad(z): return (z > 0).astype(np.float64)

def sigmoid(x):
    return expit(x)

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(x): return np.tanh(x)
def tanh_grad(z): return 1 - np.tanh(z) ** 2

#TODO add more activation functions here and in the map in constants.py - softmax