#!/bin/bash
# mpiruns.sh
# Example usage: ./mpiruns.sh

# Paths
PYTHON_SCRIPT="main.py" 

# Experiment parameters
NUM_PROCESSES=(1 2 3 4 5)
FILE_PATH="../data/processed/nytaxi2022_preprocessed_final.csv"
TEST_RATIO=0.3
HIDDEN_LAYERS=16
LEARNING_RATE=(0.0001 0.00001)
ACTIVATION_FUNCTION=("relu" "linear" "tanh" "sigmoid")
NUM_ITERATIONS=1000000
BATCH_SIZES=(32 64 128 256 512)
SEED=42

# Loop over processes and batch sizes
for P in "${NUM_PROCESSES[@]}"; do
    for B in "${BATCH_SIZES[@]}"; do
        for L in "${LEARNING_RATE[@]}"; do
            for A in "${ACTIVATION_FUNCTION[@]}"; do
                echo "Running with $P processes, batch size $B, learning rate $L, activation $A"
                mpiexec -n $P python $PYTHON_SCRIPT \
                    $FILE_PATH \
                    $TEST_RATIO \
                    $HIDDEN_LAYERS \
                    $L \
                    $A \
                    $NUM_ITERATIONS \
                    $B \
                    $SEED
                echo "--------------------------------------------"
            done
        done
    done
done
