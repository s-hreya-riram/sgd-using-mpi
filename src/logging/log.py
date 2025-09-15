import os
from processing.mpi_utils import rank

def log_metrics(epoch, batch_size, activations, test_rmse, logfile="training_log.csv"):
    """
    Appends training metrics to a CSV file. Only rank 0 writes.
    """

    if rank == 0:
        # Ensure directory exists
        os.makedirs(os.path.dirname(logfile), exist_ok=True) if os.path.dirname(logfile) else None

        # Write header if file doesn't exist
        if not os.path.exists(logfile):
            with open(logfile, "w") as f:
                f.write("epoch,batch_size,activations,test_rmse\n")

        # Append new line
        with open(logfile, "a") as f:
            f.write(f"{epoch},{batch_size},\"{','.join(activations)}\",{test_rmse:.6f}\n")
