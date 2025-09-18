import os
from processing.mpi_utils import rank

def log_test_rmse(epoch, batch_size, activation, learning_rate, train_rmse, test_rmse, logfile):
    """
    Appends training metrics to a CSV file. Only rank 0 writes.
    """

    if rank == 0:
        # Ensure directory exists
        os.makedirs(os.path.dirname(logfile), exist_ok=True) if os.path.dirname(logfile) else None

        # If file doesn't exist, then write the header, otherwise skip
        if not os.path.exists(logfile):
            with open(logfile, "w") as f:
                f.write("epoch,batch_size,activation,learning_rate,train_rmse,test_rmse\n")

        # Append new line
        with open(logfile, "a") as f:
            f.write(f"{epoch},{batch_size},{activation},{learning_rate},{train_rmse:.6f},{test_rmse:.6f}\n")

def log_training_metrics(iteration, epoch, batch_size, activation, learning_rate, training_loss, logfile="../logs/normalization_fix/training_metrics.csv"):
    """
    Appends training metrics to a CSV file. Only rank 0 writes.
    """
    if rank == 0:
        # Ensure directory exists
        logdir = os.path.dirname(logfile)
        if logdir:
            os.makedirs(logdir, exist_ok=True)

        # Check if file exists already
        file_exists = os.path.exists(logfile)

        # Open in append mode always
        with open(logfile, "a") as f:
            # If new file, write header first
            if not file_exists:
                f.write("iteration,epoch,batch_size,activation,learning_rate,training_loss\n")
            # Append new log line
            f.write(f"{iteration},{epoch},{batch_size},{activation},{learning_rate},{training_loss:.6f}\n")