#!/bin/bash

# Set the path to the Python script
python_script="train_mvit2_itodd.py"

# Set the common command
common_command="python $python_script --eval-only"

# Loop through model names and run the command
for i in {9999..369999..10000}; do
    model_name="model_$(printf %07d $i).pth"
    full_command="$common_command --model_name $model_name"
    echo "Running: $full_command"
    $full_command
done
