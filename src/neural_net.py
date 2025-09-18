# NeuralNet.py
from math import sqrt
import numpy as np
from mpi4py import MPI
from processing.mpi_utils import rank, size, comm
from utils.constants import *
from log.log import log_test_rmse, log_training_metrics


class NeuralNet:
    def __init__(self, input_dim, hidden_dim, learning_rate, activation_fn, seed):
        # randomly initialize weights by rank
        np.random.seed(seed + rank)
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        # initialize weights based on activation function
        if activation_fn == "relu": 
            # He initialization (good for ReLU) 
            self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim) 
            self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim) 
        else: 
            # default to Xavier if linear, sigmoid, tanh 
            self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim) 
            self.W2 = np.random.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, 1))


    def forward(self, X):
        """Forward pass through 1 hidden layer"""
        self.z1 = X @ self.W1 + self.b1
        activation_fn = ACTIVATION_FUNCTION_MAP[self.activation_fn][0]
        self.a1 = activation_fn(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred):
        """Compute local gradients"""
        m = X.shape[0]
        error = (y_pred - y.reshape(-1, 1))  # (m,1)

        # Gradient computation for output layer
        dW2 = self.a1.T @ error / m
        db2 = np.mean(error, axis=0, keepdims=True)

        # backpropagate the error to the hidden layer
        da1 = error @ self.W2.T
        activation_derivative = ACTIVATION_FUNCTION_MAP[self.activation_fn][1]
        dz1 = da1 * activation_derivative(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        return [dW1, db1, dW2, db2]

    def apply_gradients(self, gradients):
        """Update weights using averaged gradients"""
        dW1, db1, dW2, db2 = gradients
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2


def train_on_batch(model, X_batch, y_batch):
    """
    Distributed SGD step: forward, backward, MPI Allreduce, apply update
    """
    # Forward
    y_pred = model.forward(X_batch)

    # Compute local gradients
    local_gradients = model.backward(X_batch, y_batch, y_pred)

    # aggregate gradients across processes
    global_gradients = [np.zeros_like(local_gradient) for local_gradient in local_gradients]
    for local_gradient, global_gradient in zip(local_gradients, global_gradients):
        comm.Allreduce(local_gradient, global_gradient, op=MPI.SUM)
    global_gradients = [global_gradient / size for global_gradient in global_gradients]

    # update the weights of the model using the average global gradients
    model.apply_gradients(global_gradients)

    # Compute loss (MSE)
    loss = np.mean((y_pred - y_batch.reshape(-1, 1)) ** 2)
    return loss


def train(model, X_train, y_train, epochs, batch_size, seed):
    iteration = 0 
    loss = 0

    for epoch in range(epochs):
        # Shuffle training data (all ranks must use same permutation)
        indices = np.arange(X_train.shape[0])
        if rank == 0:
            np.random.seed(seed + epoch)
            np.random.shuffle(indices)
        comm.Bcast(indices, root=0)
        X_train, y_train = X_train[indices], y_train[indices]

        # Mini-batch loop
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Train on batch (distributed)
            loss = train_on_batch(model, X_batch, y_batch)

            # Increment iteration counter
            iteration += 1

            # Log iteration, epoch, params, and loss (only rank 0 writes)
            if rank == 0:
                log_training_metrics(iteration, epoch+1, batch_size, model.activation_fn, model.learning_rate, loss)
    return sqrt(loss)


def evaluate(model, X_test, y_test):
    """Distributed RMSE evaluation"""
    # forward pass to get predictions
    y_pred = model.forward(X_test).flatten()

    # calculating standard errors and local count
    local_sum_of_squares = float(np.sum((y_pred - y_test) ** 2))
    local_count = len(y_test)

    # Allreduce to get global sum of squared errors and count
    global_sum_of_squares = comm.allreduce(local_sum_of_squares, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    # Compute RMSE
    rmse = np.sqrt(global_sum_of_squares / global_count)
    if rank == 0:
        print(f"Test RMSE: {rmse:.4f}")
    return rmse


def execute_model(model, X_train, y_train, X_test, y_test,
                  epochs, batch_size, seed):

    # adding a barrier to ensure all ranks start timing together
    comm.Barrier()

    # tracking timing for total execution of the NN model with SGD, 
    # training time and evaluation time separately
    total_start = MPI.Wtime()
    train_start = MPI.Wtime()

    # Train and evaluate the model
    train_rmse = train(model, X_train, y_train, epochs, batch_size, seed)

    train_end = MPI.Wtime()
    eval_start = MPI.Wtime()

    test_rmse = evaluate(model, X_test, y_test)

    eval_end = MPI.Wtime()
    total_end = MPI.Wtime()

    # collect timings across ranks
    local_train_time = train_end - train_start
    local_eval_time = eval_end - eval_start
    local_total_time = total_end - total_start

    # determine the slowest time among all ranks
    train_time_max = comm.allreduce(local_train_time, op=MPI.MAX)
    eval_time_max = comm.allreduce(local_eval_time, op=MPI.MAX)
    total_time_max = comm.allreduce(local_total_time, op=MPI.MAX)

    # determine the average time across all ranks
    train_time_avg = comm.allreduce(local_train_time, op=MPI.SUM) / size
    eval_time_avg = comm.allreduce(local_eval_time, op=MPI.SUM) / size
    total_time_avg = comm.allreduce(local_total_time, op=MPI.SUM) / size

    # Log train and test RMSE
    if rank == 0:
        log_test_rmse(
            epochs, batch_size, model.activation_fn, model.learning_rate,train_rmse, test_rmse,
            logfile="../logs/normalization_fix/train_test_rmse.csv"
        )
        print(f"[Epoch {epochs}] Train RMSE: {train_rmse:.4f} Test RMSE: {test_rmse:.4f}")
        print(f"[Timing Summary] "
              f"Train: {train_time_max:.3f}s (max) / {train_time_avg:.3f}s (avg)  "
              f"Eval: {eval_time_max:.3f}s (max) / {eval_time_avg:.3f}s (avg)  "
              f"Total: {total_time_max:.3f}s (max) / {total_time_avg:.3f}s (avg)")

    return train_rmse, test_rmse
