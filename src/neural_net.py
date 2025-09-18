import numpy as np
from mpi4py import MPI
from processing.mpi_utils import rank, size, comm
from utils.constants import *
from log.log import log_test_rmse, log_training_metrics


class NeuralNet:
    """
    Defining a structure for the NeuralNet model with the parameters -
    input_dim, hidden_dim, learning_rate, activation_function, num_processes and input seed.

    We initialize the starting weights based on the activation function to ensure faster
    convergence following https://www.deeplearning.ai/ai-notes/initialization/index.html
    """
    def __init__(self, input_dim, hidden_dim, learning_rate, activation_fn, num_processes, seed):
        # randomly initialize weights by rank
        np.random.seed(seed + rank)
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.num_processes = num_processes

        # initialize weights based on activation function
        if activation_fn == "relu": 
            # He initialization (good for ReLU) 
            self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim) 
            self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim) 
        else: 
            # default to Xavier initialization if not ReLu i.e. linear, sigmoid, tanh 
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
    For a given batch of training data, we do the following steps:
     - calculate y_pred for train data (batch) using forward propagation, 
     - backpropagate error between y_pred and y_train (batch) to get the local gradients 
     - execute MPI allreduce to sum the local gradients to get the global gradients, 
     - apply gradient updates to adjust the weights of the model
     - compute and return the local sum of squared errors for the global train RMSE calculation in train()
    """
    batch_size = X_batch.shape[0]
    rank_batch_size = int(np.ceil(batch_size / size))  # splitting nearly equally across ranks

    # Determine slice for this rank
    begin_index_local = rank * rank_batch_size
    end_index_local = min((rank + 1) * rank_batch_size, batch_size)
    X_local = X_batch[begin_index_local:end_index_local]
    y_local = y_batch[begin_index_local:end_index_local]

    if X_local.shape[0] > 0:
        # Forward propagation on local slice
        y_pred_local = model.forward(X_local)
        # Compute local gradients
        local_gradients = model.backward(X_local, y_local, y_pred_local)
    else:
        # Skipping forward and backward propagation if no data for this rank
        local_gradients = [
            np.zeros_like(model.W1),
            np.zeros_like(model.b1),
            np.zeros_like(model.W2),
            np.zeros_like(model.b2)
        ]
        y_pred_local = np.zeros((0,1))

    # Aggregate gradients across ranks (average)
    global_gradients = [comm.allreduce(g, op=MPI.SUM) / size for g in local_gradients]

    # update the weights of the model using the average global gradients
    model.apply_gradients(global_gradients)

    # Compute local sum of squared errors for RMSE calculation
    local_sum_of_squares = np.sum((y_pred_local - y_local.reshape(-1, 1)) ** 2)
    return local_sum_of_squares

def train_on_batch(model, X_batch, y_batch):
    """
    For a given batch of training data, we do the following steps:
     - calculate y_pred for train data (batch) using forward propagation, 
     - backpropagate error between y_pred and y_train (batch) to get the local gradients 
     - execute MPI allreduce to sum the local gradients to get the global gradients, 
     - apply gradient updates to adjust the weights of the model
     - compute and return the local sum of squared errors for the global train RMSE calculation in train()
    """
    batch_size = X_batch.shape[0]
    rank_batch_size = int(np.ceil(batch_size / size))  # splitting nearly equally across ranks

    # Determine slice for this rank
    begin_index_local = rank * rank_batch_size
    end_index_local = min((rank + 1) * rank_batch_size, batch_size)
    X_local = X_batch[begin_index_local:end_index_local]
    y_local = y_batch[begin_index_local:end_index_local]

    if X_local.shape[0] > 0:
        # Forward propagation on local slice
        y_pred_local = model.forward(X_local)
        # Compute local gradients
        local_gradients = model.backward(X_local, y_local, y_pred_local)
    else:
        # Skipping forward and backward propagation if no data for this rank
        local_gradients = [
            np.zeros_like(model.W1),
            np.zeros_like(model.b1),
            np.zeros_like(model.W2),
            np.zeros_like(model.b2)
        ]
        y_pred_local = np.zeros((0,1))

    # get number of samples on this rank
    num_samples_local = X_local.shape[0]

    # TODO revisit if weighing is a good idea
    # sum gradients weighted by local sample size to account for uneven distribution
    weighted_local_gradients = [local_gradient * num_samples_local for local_gradient in local_gradients]

    # Aggregate gradients across ranks using MPI SUM
    summed_gradients = [comm.allreduce(g, op=MPI.SUM) for g in weighted_local_gradients]

    # divide by total number of samples contributing
    total_samples = comm.allreduce(num_samples_local, op=MPI.SUM)
    global_gradients = [g / total_samples for g in summed_gradients]

    # update the weights of the model using the average global gradients
    model.apply_gradients(global_gradients)

    # Compute local sum of squared errors for RMSE calculation
    local_sum_of_squares = np.sum((y_pred_local - y_local.reshape(-1, 1)) ** 2)
    return local_sum_of_squares



def train(model, X_train, y_train, epochs, batch_size, seed):
    iteration = 0 
    total_sum_of_squares = 0.0
    total_count = 0

    for epoch in range(epochs):
        # shuffle training data uniformly across all processes
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

            # Train on batch (slicing the batch across processes within train_on_batch)
            local_sum_of_squares = train_on_batch(model, X_batch, y_batch)

            # Increment iteration counter
            iteration += 1

            global_sum_of_square_errors = comm.allreduce(local_sum_of_squares, op=MPI.SUM)
            global_count = comm.allreduce(len(y_batch), op=MPI.SUM)
            total_sum_of_squares += global_sum_of_square_errors
            total_count += global_count
            batch_loss = global_sum_of_square_errors / global_count

            # Log iteration, epoch, params, and loss (only rank 0 writes)
            if rank == 0:
                log_training_metrics(iteration, epoch+1, batch_size, model.activation_fn, model.learning_rate, model.num_processes, batch_loss)


    # returning train RMSE as sqrt(training loss)
    return np.sqrt(total_sum_of_squares / total_count)


def evaluate(model, X_test, y_test):
    """The evaluation function does the following:
    - performs forward propagation to get y_pred for the feature data
    - computes the RMSE on the test data using y_pred and y_test.
    Returns the test RMSE"""
    # forward pass to get predictions
    y_pred = model.forward(X_test).flatten()

    # calculating standard errors and local count
    local_sum_of_squares = float(np.sum((y_pred - y_test) ** 2))
    local_count = len(y_test)

    # allreduce to get global sum of squared errors and count
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
    test_start = MPI.Wtime()

    test_rmse = evaluate(model, X_test, y_test)

    test_end = MPI.Wtime()
    total_end = MPI.Wtime()

    # collect timings across ranks
    local_train_time = train_end - train_start
    local_test_time = test_end - test_start
    local_total_time = total_end - total_start

    # determine the slowest time among all ranks
    train_time_max = comm.allreduce(local_train_time, op=MPI.MAX)
    test_time_max = comm.allreduce(local_test_time, op=MPI.MAX)
    total_time_max = comm.allreduce(local_total_time, op=MPI.MAX)

    # determine the average time across all ranks
    train_time_avg = comm.allreduce(local_train_time, op=MPI.SUM) / size
    test_time_avg = comm.allreduce(local_test_time, op=MPI.SUM) / size
    total_time_avg = comm.allreduce(local_total_time, op=MPI.SUM) / size

    # Log train and test RMSE
    if rank == 0:
        log_test_rmse(
            model.num_processes, epochs, batch_size, model.activation_fn, model.learning_rate,
            train_rmse, test_rmse, train_time_max, train_time_avg, test_time_max, test_time_avg, 
            total_time_max, total_time_avg,
            logfile="../logs/normalization_fix/train_test_rmse.csv"
        )
        print(f"[Epoch {epochs}] Train RMSE: {train_rmse:.4f} Test RMSE: {test_rmse:.4f}")
        print(f"[Timing Summary] "
              f"Train: {train_time_max:.3f}s (max) / {train_time_avg:.3f}s (avg)  "
              f"Test: {test_time_max:.3f}s (max) / {test_time_avg:.3f}s (avg)  "
              f"Total: {total_time_max:.3f}s (max) / {total_time_avg:.3f}s (avg)")

    return train_rmse, test_rmse
