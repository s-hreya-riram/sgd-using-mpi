import numpy as np

def linear(x): return x
def linear_grad(z): return np.ones_like(z)

def relu(x): return np.maximum(0, x)
def relu_grad(z): return (z > 0).astype(np.float64)

def sigmoid(x):
    return np.where(x>=0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(x): return np.tanh(x)
def tanh_grad(z): return 1 - np.tanh(z) ** 2
