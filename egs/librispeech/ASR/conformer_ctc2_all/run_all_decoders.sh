#!/bin/bash
### jz3702
echo "Running all models for decoding to get WERs"

# Change to the directory where the subdirectories are located
# cd ./exp

# Iterate over each subdirectory
for dir in exp/*/; do
    # Remove the trailing slash to get the directory name
    dir_name=${dir%/}

    # Print the directory name
    echo "Running with exp-dir: $dir_name"

    # Run the python command with the current directory
    python decode.py \
        --epoch 30 \
        --avg 1  \
        --max-duration 150   \
        --exp-dir "$PWD/$dir_name"  \
        --lang-dir ../data/lang_bpe_500  \
        --method ctc-decoding
done

# Change back to the original directory
cd ../..