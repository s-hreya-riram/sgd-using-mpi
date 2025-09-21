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

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
def leaky_relu_grad(z, alpha=0.01):
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
def elu_grad(z, alpha=1.0):
    dz = np.ones_like(z)
    dz[z < 0] = elu(z[z < 0], alpha) + alpha
    return dz