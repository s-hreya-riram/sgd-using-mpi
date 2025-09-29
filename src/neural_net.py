from math import sqrt
import numpy as np
from mpi4py import MPI
from processing.mpi_utils import rank, size, comm
from utils.constants import *
from log.log import log_test_rmse, log_training_metrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NeuralNet:
    """
    Defining a structure for the NeuralNet model with the parameters -
    input_dim, hidden_dim, learning_rate, activation_function, num_processes and input seed.

    Below the starting weights are initialized based on the activation function to ensure faster
    convergence following https://www.deeplearning.ai/ai-notes/initialization/index.html
    """
    def __init__(self, input_dim, hidden_dim, learning_rate, activation_fn, num_processes, seed, debug=False):
        logger.info(f"Rank {rank} initializing NeuralNet...")
        # randomly initialize weights by rank
        rng = np.random.default_rng(seed + rank)
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.num_processes = num_processes
        self.debug = debug

        if self.debug:
            logger.setLevel(logging.DEBUG)

        # initialize weights based on activation function
        # w1 and b1 are weights and biases for the hidden layer
        # w2 and b2 are weights and biases for the output layer
        if activation_fn == "relu":
            # He initialization (good for ReLU)
            self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
            self.W2 = rng.standard_normal((hidden_dim, 1)) * np.sqrt(2.0 / hidden_dim)
        else:
            # default to Xavier initialization if not ReLu i.e. linear, sigmoid, tanh
            self.W1 = rng.standard_normal((input_dim, hidden_dim)) / np.sqrt(input_dim)
            self.W2 = rng.standard_normal((hidden_dim, 1)) / np.sqrt(hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, 1))
        # Broadcast initial weights from rank 0 to all other ranks
        comm.Bcast(self.W1, root=0)
        comm.Bcast(self.b1, root=0)
        comm.Bcast(self.W2, root=0)
        comm.Bcast(self.b2, root=0)
        logger.info(f"Rank {rank} initialized weights and biases.")

    def forward(self, X):
        logger.info(f"Rank {rank} performing forward pass")
        """Forward pass through 1 hidden layer"""
        self.z1 = X @ self.W1 + self.b1
        activation_fn = ACTIVATION_FUNCTION_MAP[self.activation_fn][0]
        self.a1 = activation_fn(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred):
        """Compute local gradients with backward pass through 1 hidden layer"""
        logger.info(f"Rank {rank} performing backward pass")
        batch_size = X.shape[0]
        error = (y_pred - y.reshape(-1, 1))  # (batch_size,1)

        # Gradient computation for output layer
        dW2 = self.a1.T @ error / batch_size
        db2 = np.mean(error, axis=0, keepdims=True)

        # backpropagate the error to the hidden layer
        da1 = error @ self.W2.T
        activation_derivative = ACTIVATION_FUNCTION_MAP[self.activation_fn][1]
        dz1 = da1 * activation_derivative(self.z1)
        dW1 = X.T @ dz1 / batch_size
        db1 = np.mean(dz1, axis=0, keepdims=True)

        return [dW1, db1, dW2, db2]

    def apply_gradients(self, gradients):
        """Update weights using averaged gradients"""
        logger.info(f"Rank {rank} applying gradients")
        dW1, db1, dW2, db2 = gradients
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

def cyclical_lr(iteration, base_lr=1e-5, step_size=500, max_lr=2e-4):
    """
    Triangular cyclical learning rate with cycle length varying based on 
    batch size (through step_size) and number of processes (noted as size).
    - base_lr: minimum LR
    - max_lr: maximum LR
    - step_size: number of iterations to reach max_lr from base_lr.
    """
    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs(iteration / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * np.maximum(0, 1 - x) * size

def train_on_batch(model, X_local, y_local):
    """
    Train on a local batch and return local SSE.
    - Execute allreduce for gradients
    - Handle weighted gradients for uneven batch sizes
    - Return local SSE for train RMSE calculation
    """
    num_local = X_local.shape[0]

    if num_local > 0:
        # Forward and backward pass
        y_pred_local = model.forward(X_local)
        local_gradients = model.backward(X_local, y_local, y_pred_local)

        # Weight gradients by number of local samples as samples per rank may differ
        weighted_gradients = [g * num_local for g in local_gradients]
    else:
        # fallback for ranks with no data in batch
        y_pred_local = np.zeros((0, 1))
        weighted_gradients = [
            np.zeros_like(model.W1),
            np.zeros_like(model.b1),
            np.zeros_like(model.W2),
            np.zeros_like(model.b2)
        ]

    # Aggregate gradients across all processes
    summed_gradients = [comm.allreduce(g, op=MPI.SUM) for g in weighted_gradients]
    total_samples = comm.allreduce(num_local, op=MPI.SUM)

    if total_samples > 0:
        global_gradients = [sg / total_samples for sg in summed_gradients]
    else:
        global_gradients = [np.zeros_like(g) for g in weighted_gradients]

    # Apply global gradients
    model.apply_gradients(global_gradients)

    # Compute local SSE
    local_sse = float(np.sum((y_pred_local - y_local.reshape(-1, 1)) ** 2)) if num_local > 0 else 0.0
    return local_sse

def train(model, X_train, y_train, max_iterations, batch_size, seed, stopping_criterion):
    """
    Distributed SGD training loop using mini-batch approximation for convergence.
    - Each iteration computes gradients using a batch of randomly selected samples.
    - Convergence is checked using batch-level loss.
    - Final RMSE is computed on the full training set.
    """
    rng = np.random.default_rng(seed + rank)
    iteration = 0
    previous_loss = None
    base_lr = model.learning_rate

    while iteration < max_iterations:
        # Sample M indices for this iteration (mini-batch)
        indices = rng.choice(X_train.shape[0], batch_size, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        # Update learning rate using a cyclic scheduler
        # defining step sizes based on batch size to reduce training time
        if batch_size >= 256:
            step_size = 1000
        elif batch_size >= 128:
            step_size = 2000
        else:
            step_size = 2500
        model.learning_rate = cyclical_lr(iteration, step_size=step_size, base_lr=base_lr)

        # Train on batch and get batch SSE
        local_sse = train_on_batch(model, X_batch, y_batch)
        local_count = X_batch.shape[0]

        # Compute batch-level global loss for convergence
        global_sse = comm.allreduce(local_sse, op=MPI.SUM)
        global_count = comm.allreduce(local_count, op=MPI.SUM)
        current_loss = 0.5 * (global_sse / global_count) if global_count > 0 else 0.0

        # Logging
        if rank == 0:
            log_training_metrics(iteration, batch_size,
                                 model.activation_fn, model.learning_rate,
                                 model.num_processes, current_loss)
            logger.debug(f"Iteration {iteration}, batch-loss={current_loss:.6f}")

        # Check convergence using batch-level loss
        if previous_loss is not None and abs(previous_loss - current_loss) / previous_loss < stopping_criterion:
            if rank == 0:
                logger.info(f"Converged at iteration {iteration}, batch-loss={current_loss:.6f}")
            break

        previous_loss = current_loss
        iteration += 1

    # Compute final RMSE on the full training set
    y_pred_full = model.forward(X_train)
    local_sse_full = np.sum((y_pred_full - y_train.reshape(-1, 1)) ** 2)
    global_sse_full = comm.allreduce(local_sse_full, op=MPI.SUM)
    global_count_full = comm.allreduce(X_train.shape[0], op=MPI.SUM)
    train_rmse = np.sqrt(global_sse_full / global_count_full)

    return train_rmse, iteration

def evaluate(model, X_test, y_test):
    """Execute evaluation on the test set and return RMSE."""
    logger.debug(f"Rank {rank} starting evaluation on test data of size {X_test.shape[0]}")
    y_pred = model.forward(X_test).flatten()
    local_sse = float(np.sum((y_pred - y_test) ** 2))
    local_count = len(y_test)

    global_sse = comm.allreduce(local_sse, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    rmse = np.sqrt(global_sse / global_count)
    if rank == 0:
        logger.info(f"Test RMSE: {rmse:.4f}")
    logger.debug(f"Rank {rank} completed evaluation")
    return rmse


def execute_model(model, X_train, y_train, X_test, y_test,
                  max_iterations, batch_size, seed):
    """
    Executes distributed training + evaluation with timing and logging.
    """
    stopping_criterion = 1e-6
    logger.debug(f"Rank {rank} executing model training and evaluation")
    comm.Barrier()

    total_start = MPI.Wtime()
    train_start = MPI.Wtime()

    # Train the model and get the training RMSE for comparison
    train_rmse, num_iterations = train(model, X_train, y_train, max_iterations, batch_size, seed, stopping_criterion)

    train_end = MPI.Wtime()
    eval_start = MPI.Wtime()

    # Evaluate on test set
    test_rmse = evaluate(model, X_test, y_test)

    eval_end = MPI.Wtime()
    total_end = MPI.Wtime()

    # Measure times (local per rank)
    local_train_time = train_end - train_start
    local_eval_time = eval_end - eval_start
    local_total_time = total_end - total_start

    # Get max/avg timings across ranks
    train_time_max = comm.allreduce(local_train_time, op=MPI.MAX)
    eval_time_max = comm.allreduce(local_eval_time, op=MPI.MAX)
    total_time_max = comm.allreduce(local_total_time, op=MPI.MAX)

    train_time_avg = comm.allreduce(local_train_time, op=MPI.SUM) / size
    eval_time_avg = comm.allreduce(local_eval_time, op=MPI.SUM) / size
    total_time_avg = comm.allreduce(local_total_time, op=MPI.SUM) / size

    # Log only from rank 0
    if rank == 0:
        log_test_rmse(
            model.num_processes, num_iterations, 
            max_iterations, batch_size,
            model.activation_fn, model.learning_rate,
            train_rmse, test_rmse,
            train_time_max, train_time_avg,
            eval_time_max, eval_time_avg,
            total_time_max, total_time_avg,
            logfile="../logs/normalization_fix/final/train_test_rmse.csv"
        )
        logger.info(f"[Final Results] Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        logger.info(f"[Timing Summary] "
                    f"Train: {train_time_max:.3f}s (max) / {train_time_avg:.3f}s (avg)  "
                    f"Test: {eval_time_max:.3f}s (max) / {eval_time_avg:.3f}s (avg)  "
                    f"Total: {total_time_max:.3f}s (max) / {total_time_avg:.3f}s (avg)")

    logger.info(f"Rank {rank} finished model execution")
    return train_rmse, test_rmse
