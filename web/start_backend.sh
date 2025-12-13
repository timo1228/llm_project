#!/bin/bash

# Start backend server
cd "$(dirname "$0")/backend"
conda activate image-upload
python main.py



