#!/bin/bash
### jz3702
echo "Run all models for training"

# Define the list of activation types
activation_types=('tanh' 'softplus' 'relu' 'gelu' 'leaky_relu' 'elu' 'selu' 'swish' 'double_swish')

# Get the current date in the format YYYYMMDD
current_date=$(date +"%Y%m%d")

# Iterate over each activation type
for activation_type in "${activation_types[@]}"; do
    # Construct the experiment directory path
    exp_dir="exp/run_${current_date}_${activation_type}_clean/"

    # Run the python command with the current activation type and experiment directory
    python train.py \
        --exp-dir "$exp_dir" \
        --start-epoch 1 \
        --world-size 4 \
        --full-libri 0 \
        --max-duration 250 \
        --activation_type "$activation_type" \
        --all_activation no
done