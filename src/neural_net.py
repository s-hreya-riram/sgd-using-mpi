# NeuralNet.py
from math import sqrt
import numpy as np
from mpi4py import MPI
from processing.mpi_utils import rank, size, comm
from utils.constants import *
from log.log import log_test_rmse, log_training_metrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed internal state logs

class NeuralNet:
    """
    Defining a structure for the NeuralNet model with the parameters -
    input_dim, hidden_dim, learning_rate, activation_function, num_processes and input seed.

    We initialize the starting weights based on the activation function to ensure faster
    convergence following https://www.deeplearning.ai/ai-notes/initialization/index.html
    """
    def __init__(self, input_dim, hidden_dim, learning_rate, activation_fn, num_processes, seed):
        logger.info(f"Rank {rank} initializing NeuralNet...")
        # randomly initialize weights by rank
        np.random.seed(seed + rank)
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.num_processes = num_processes

        # initialize weights based on activation function
        # w1 and b1 are weights and biases for the hidden layer
        # w2 and b2 are weights and biases for the output layer
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
    Distributed SGD training loop with:
    - Random mini-batch sampling without replacement to prevent bias
    - Convergence-based stopping criterion (global loss delta < stopping_criterion)
    - Capped by max iterations if the stopping criterion is not met
    """
    logger.debug(f"Rank {rank} starting training with batch size {batch_size}")

    iteration = 0
    total_sse = 0.0
    total_count = 0
    previous_loss = None
    rng = np.random.default_rng(seed + rank)

    while iteration < max_iterations:
        # Randomly sample batch
        indices = rng.choice(X_train.shape[0], batch_size, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        # Train on this batch
        local_sse = train_on_batch(model, X_batch, y_batch)
        local_count = X_batch.shape[0]

        # Update totals
        total_sse += local_sse
        total_count += local_count
        iteration += 1

        # Compute global loss
        global_sse = comm.allreduce(local_sse, op=MPI.SUM)
        global_count = comm.allreduce(local_count, op=MPI.SUM)
        current_loss = 0.5 * (global_sse / global_count) if global_count > 0 else 0.0

        # Log metrics
        if rank == 0:
            log_training_metrics(iteration, batch_size,
                                 model.activation_fn, model.learning_rate,
                                 model.num_processes, current_loss)
            logger.debug(f"Iteration {iteration}, loss={current_loss:.6f}")

        # Check convergence
        if previous_loss is not None and abs(previous_loss - current_loss) / previous_loss < stopping_criterion:
            if rank == 0:
                logger.info(f"Converged at iteration {iteration}, loss={current_loss:.6f}")
            return np.sqrt(global_sse / global_count) if global_count > 0 else 0.0, iteration

        previous_loss = current_loss

    # Final RMSE
    global_sse = comm.allreduce(total_sse, op=MPI.SUM)
    global_count = comm.allreduce(total_count, op=MPI.SUM)
    return np.sqrt(global_sse / global_count) if global_count > 0 else 0.0, iteration


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
    stopping_criterion = 1e-5
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
            logfile="../logs/normalization_fix/new/train_test_rmse.csv"
        )
        logger.info(f"[Final Results] Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        logger.info(f"[Timing Summary] "
                    f"Train: {train_time_max:.3f}s (max) / {train_time_avg:.3f}s (avg)  "
                    f"Test: {eval_time_max:.3f}s (max) / {eval_time_avg:.3f}s (avg)  "
                    f"Total: {total_time_max:.3f}s (max) / {total_time_avg:.3f}s (avg)")

    logger.info(f"Rank {rank} finished model execution")
    return train_rmse, test_rmse
