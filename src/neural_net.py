# NeuralNet.py
import numpy as np
from mpi4py import MPI
from processing.mpi_utils import rank, size, comm
from utils.constants import *
from log.log import log_test_rmse, log_training_metrics


class NeuralNet:
    def __init__(self, input_dim, hidden_dim=16, lr=0.01, seed=42):
        np.random.seed(seed + rank)  # different init per rank
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    # ---------------- Forward ----------------
    def forward(self, X, activation):
        """Forward pass through 1 hidden layer"""
        self.z1 = X @ self.W1 + self.b1
        activation_fn = ACTIVATION_FUNCTION_MAP[activation][0]
        self.a1 = activation_fn(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2  # linear output

    # ---------------- Backward ----------------
    def backward(self, X, y, y_pred, activation):
        """Compute local gradients"""
        m = X.shape[0]
        error = (y_pred - y.reshape(-1, 1))  # (m,1)

        # Grad for output layer
        dW2 = self.a1.T @ error / m
        db2 = np.mean(error, axis=0, keepdims=True)

        # Backprop hidden
        da1 = error @ self.W2.T
        activation_prime = ACTIVATION_FUNCTION_MAP[activation][1]
        dz1 = da1 * activation_prime(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        return [dW1, db1, dW2, db2]

    # ---------------- Gradient Apply ----------------
    def apply_gradients(self, grads):
        """Update weights using averaged gradients"""
        dW1, db1, dW2, db2 = grads
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


# =====================================================
# Training Functions
# =====================================================
def train_on_batch(model, X_batch, y_batch, activation):
    """
    Distributed SGD step: forward, backward, MPI Allreduce, apply update
    """
    # Forward
    y_pred = model.forward(X_batch, activation)

    # Compute local gradients
    local_grads = model.backward(X_batch, y_batch, y_pred, activation)

    # Allocate space for global gradients
    global_grads = [np.zeros_like(g) for g in local_grads]

    # Allreduce to aggregate gradients across processes
    for g_local, g_global in zip(local_grads, global_grads):
        comm.Allreduce(g_local, g_global, op=MPI.SUM)

    # Average across processes
    global_grads = [g / size for g in global_grads]

    # Apply update
    model.apply_gradients(global_grads)

    # Compute loss (MSE)
    loss = np.mean((y_pred - y_batch.reshape(-1, 1)) ** 2)
    return loss


def train(model, X_train, y_train, activation, epochs=10, batch_size=256, shuffle_seed=42):
    iteration = 0  # k counter

    for epoch in range(epochs):
        # Shuffle training data (all ranks must use same permutation)
        indices = np.arange(X_train.shape[0])
        if rank == 0:
            np.random.seed(shuffle_seed + epoch)
            np.random.shuffle(indices)
        comm.Bcast(indices, root=0)
        X_train, y_train = X_train[indices], y_train[indices]

        # Mini-batch loop
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Train on batch (distributed)
            loss = train_on_batch(model, X_batch, y_batch, activation)

            # Increment iteration counter
            iteration += 1

            # Log iteration, epoch, params, and loss (only rank 0 writes)
            if rank == 0:
                log_training_metrics(iteration, epoch+1, batch_size, activation, loss)


def evaluate(model, X_test, y_test, activation):
    """Distributed RMSE evaluation"""
    y_pred = model.forward(X_test, activation).flatten()
    local_se = np.sum((y_pred - y_test) ** 2)
    local_count = len(y_test)

    # Allreduce sum of squared errors
    global_se = comm.allreduce(local_se, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    rmse = np.sqrt(global_se / global_count)
    if rank == 0:
        print(f"Test RMSE: {rmse:.4f}")
    return rmse


def execute_model(model, X_train, y_train, X_test, y_test,
                  activation, epochs=5, batch_size=32, shuffle_seed=42):
    """Full pipeline: train + evaluate + log"""
    train(model, X_train, y_train, activation, epochs, batch_size, shuffle_seed)
    test_rmse = evaluate(model, X_test, y_test, activation)

    # Log the results (only once)
    if rank == 0:
        log_test_rmse(epochs, batch_size, activation, test_rmse,
                      logfile="../logs/test_rmse.csv")
        print(f"[Epoch {epochs}] Test RMSE: {test_rmse:.4f}")
