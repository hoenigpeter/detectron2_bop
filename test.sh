#!/bin/bash

# Set the path to the Python script
python_script="frcnn_retnet_base_auto_test.py"

# Set the common command
common_command="python $python_script --mode eval --model_type retinanet --dataset_config dataset_configs/itodd_random_val.json --aug"

# Loop through model names and run the command
for i in {9999..369999..10000}; do
    model_name="model_$(printf %07d $i).pth"
    full_command="$common_command --model_name $model_name"
    echo "Running: $full_command"
    $full_command
done
