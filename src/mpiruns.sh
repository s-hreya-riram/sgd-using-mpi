#!/bin/bash
# mpiruns.sh
# Example usage: ./mpiruns.sh

# Paths
PYTHON_SCRIPT="main.py" 

# Experiment parameters
NUM_PROCESSES=(5)
FILE_PATH="../data/processed/nytaxi2022_preprocessed_final.csv"
TEST_RATIO=0.3
HIDDEN_LAYERS=16
LEARNING_RATE=0.00001
ACTIVATION_FUNCTION=("relu")
NUM_ITERATIONS=1000000
BATCH_SIZES=(16 32)
SEED=123
DEBUG_MODE=y  # Set to true to enable debug mode

# Loop over processes and batch sizes
for P in "${NUM_PROCESSES[@]}"; do
    for B in "${BATCH_SIZES[@]}"; do
        for A in "${ACTIVATION_FUNCTION[@]}"; do
            echo "Running with $P processes, batch size $B, learning rate $L, activation $A"
            mpiexec -n $P python $PYTHON_SCRIPT \
                $FILE_PATH \
                $TEST_RATIO \
                $HIDDEN_LAYERS \
                $LEARNING_RATE \
                $A \
                $NUM_ITERATIONS \
                $B \
                $SEED \
                $DEBUG_MODE
            echo "--------------------------------------------"
        done
    done
done
