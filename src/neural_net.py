# NeuralNet.py
import numpy as np
from mpi4py import MPI
from processing.mpi_utils import rank, size, comm
from utils.constants import *
from utils.activation_functions import relu, relu_grad

class NeuralNet:
    def __init__(self, input_dim, hidden_dim=16, lr=0.01, seed=42):
        np.random.seed(seed + rank)  # different init per rank
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2  # linear output

    def backward(self, X, y, y_pred):
        m = X.shape[0]
        error = (y_pred - y.reshape(-1, 1))  # (m,1)

        # Grad for output layer
        dW2 = self.a1.T @ error / m
        db2 = np.mean(error, axis=0, keepdims=True)

        # Backprop hidden
        da1 = error @ self.W2.T
        dz1 = da1 * relu_grad(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, grads):
        dW1, db1, dW2, db2 = grads
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


def train(model, X_train, y_train, epochs=5, batch_size=32, shuffle_seed=42):
    np.random.seed(shuffle_seed + rank)
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for batch_idx, start in enumerate(range(0, n_samples, batch_size)):
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            if X_batch.shape[0] == 0:
                continue

            y_pred = model.forward(X_batch)
            grads = model.backward(X_batch, y_batch, y_pred)

            # Average gradients across ranks
            averaged_grads = []
            for g in grads:
                g_avg = np.zeros_like(g)
                comm.Allreduce(g, g_avg, op=MPI.SUM)
                g_avg /= size
                averaged_grads.append(g_avg)

            model.update(averaged_grads)

            # Debug logs: only for first batch per epoch
            if batch_idx == 0:
                local_loss = np.mean((y_pred.flatten() - y_batch) ** 2)

                # Allreduce for scalar loss
                local_loss_arr = np.array([local_loss], dtype=np.float64)
                global_loss_arr = np.zeros(1, dtype=np.float64)
                comm.Allreduce(local_loss_arr, global_loss_arr, op=MPI.SUM)
                global_loss = global_loss_arr[0] / size

                print(f"[Rank {rank}] Epoch {epoch+1}, Batch {batch_idx+1}: "
                      f"Local Loss={local_loss:.4f}, Global Loss={global_loss:.4f}")

                # Show predictions vs true values for first few samples
                if rank == 0:
                    for i in range(min(3, len(y_batch))):
                        print(f"[DEBUG] Sample {i}: pred={y_pred[i,0]:.4f}, true={y_batch[i]:.4f}")

                    print(f"[DEBUG] W1 mean={np.mean(model.W1):.6f}, "
                          f"W2 mean={np.mean(model.W2):.6f}")

        if rank == 0:
            print(f"[Epoch {epoch+1}] training step done")


def evaluate(model, X_test, y_test):
    y_pred = model.forward(X_test).flatten()
    local_se = np.sum((y_pred - y_test) ** 2)
    local_count = len(y_test)

    # Allreduce for squared error
    local_se_arr = np.array([local_se], dtype=np.float64)
    global_se_arr = np.zeros(1, dtype=np.float64)
    comm.Allreduce(local_se_arr, global_se_arr, op=MPI.SUM)
    global_se = global_se_arr[0]

    # Allreduce for count
    local_count_arr = np.array([local_count], dtype=np.int64)
    global_count_arr = np.zeros(1, dtype=np.int64)
    comm.Allreduce(local_count_arr, global_count_arr, op=MPI.SUM)
    global_count = global_count_arr[0]

    rmse = np.sqrt(global_se / global_count)
    if rank == 0:
        print(f"Test RMSE: {rmse:.4f}")
    return rmse