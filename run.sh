#!/bin/bash

# Exit script on any error
# set -e

# Activate the virtual environment (if any)
# Uncomment and adjust the line below if you're using a virtual environment
# source venv/bin/activate
conda activate codeopt

# Set environment variables (if needed)
# export ENV_VAR_NAME="value"

# Paths to datasets
TRAIN_DATA_PATH="processed_data/python/processed_train.jsonl"
TEST_DATA_PATH="processed_data/python/processed_test.jsonl"
OUTPUT_PATH="outputs"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"

# Command-line arguments
MODE="optimize"
DEVICE="mps"  # 'mps' is for apple chips. Use 'cuda' for GPU or 'cpu' for CPU

CONFIG_PATH="configs/files/codellama_config.json"

# Run the main script
python src/main.py \
  --mode $MODE \
  --train_data_path $TRAIN_DATA_PATH \
  --test_data_path $TEST_DATA_PATH \
  --output_path $OUTPUT_PATH \
  --model_name $MODEL_NAME \
  --device $DEVICE \
  --config_path $CONFIG_PATH