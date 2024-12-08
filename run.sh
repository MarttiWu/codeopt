#!/bin/bash

conda activate codeopt

# Paths to datasets and model
CONFIG_PATH="configs/mistral_config.json"

# Run the main script
python src/main.py --config_path $CONFIG_PATH