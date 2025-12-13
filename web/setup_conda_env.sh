#!/bin/bash

# Create conda environment for the project
conda create -n image-upload python=3.10 -y

# Activate environment
conda activate image-upload

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies (requires Node.js)
cd ../frontend
npm install

echo "Environment setup complete!"
echo ""
echo "To start the backend server:"
echo "  conda activate image-upload"
echo "  cd backend"
echo "  python main.py"
echo ""
echo "To start the frontend (in a new terminal):"
echo "  cd frontend"
echo "  npm run dev"



