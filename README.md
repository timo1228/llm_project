# NutriVision

**A Parameter-Efficient LMM for Food Image-to-Nutrition Analysis**

NutriVision is a web application that uses a parameter-efficient Large Multimodal Model (LMM) to analyze food images and generate detailed nutrition reports. The system is built on Qwen2.5-VL-7B-Instruct with LoRA fine-tuning for efficient adaptation to nutrition analysis tasks.

## Project Overview

This project consists of:
- **Frontend**: React-based web interface for image upload and result display
- **Backend**: FastAPI server with Qwen2.5-VL-7B-Instruct model and LoRA adapter
- **Model Training**: Fine-tuning scripts and utilities for Qwen2.5-VL model
- **Data**: Training data and processed nutrition reports

## Project Structure

```
Project/
├── frontend/              # React frontend application (NutriVision UI)
│   ├── src/               # Source code for the UI
│   ├── package.json       # Node.js dependencies
│   ├── vite.config.js     # Build configuration
│   └── README.md          # Frontend specific documentation (See for more details)
│
├── web/                   # Web backend services
│   └── backend/           # FastAPI backend server
│       ├── main.py        # API entry point
│       ├── requirements.txt # Python dependencies for the backend
│       └── README.md      # Backend documentation (See for API details and configuration)
│
├── Qwen2.5-VL-Finetune/   # Model fine-tuning module
│   ├── train.py           # Main training script for LoRA fine-tuning
│   ├── eval.ipynb         # Notebook for model evaluation
│   ├── qwen-vl-finetune/  # Core fine-tuning utilities and configs
│   └── README.md          # Fine-tuning documentation (See for training steps)
│
├── blip2/                 # BLIP-2 model exploration module
│   ├── custom/            # Custom BLIP-2 model implementations
│   ├── pretrain_eval.ipynb# Pre-training evaluation notebook
│   └── blip2.ipynb        # BLIP-2 experiments notebook
│
├── data/                  # Dataset management
│   ├── nutrition5k_dataset/ # Raw Nutrition5k dataset (downloaded)
│   ├── processed/         # Processed dataset ready for training
│   ├── image_report_pairs.json # Image-text pairs for training
│   └── validate_and_split_data.py # Script to validate and split data
│
├── scripts/               # Utility scripts for evaluation and data processing
│   ├── evaluate_rouge.py  # Script to calculate ROUGE scores
│   ├── create_pairs.py    # Helper to create image-text pairs
│   └── generate_reports.py# Batch generation script
│
├── setup_conda_env.sh     # Bash script to set up the Conda environment
└── README.md              # This file
```

> **Note**: Each module (frontend, backend, Qwen2.5-VL-Finetune) contains its own `README.md` with detailed instructions specific to that component. Please refer to them for in-depth setup and usage guides.

## Features

- 🖼️ **Image Upload**: Easy-to-use interface for uploading food images
- 🤖 **AI-Powered Analysis**: Advanced LMM for accurate nutrition analysis
- 📊 **Detailed Reports**: Comprehensive nutrition reports with ingredient breakdown
- ⚡ **Efficient Processing**: Parameter-efficient LoRA fine-tuning for faster inference
- 🎨 **Modern UI**: Clean, responsive interface built with React and Tailwind CSS

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- CUDA-capable GPU (optional, for faster inference)
- Conda (recommended for environment management)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/timo1228/llm_project.git
   cd llm_project
   ```

2. **Set up the environment**:
   ```bash
   bash setup_conda_env.sh
   ```

3. **Install backend dependencies**:
   ```bash
   conda activate image-upload
   cd web/backend
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**:
   ```bash
   cd ../../frontend
   npm install
   ```

### Running the Application

1. **Start the backend server** (Terminal 1):
   ```bash
   conda activate image-upload
   cd web/backend
   python main.py
   ```
   Backend will run at `http://localhost:8000`.

2. **Start the frontend development server** (Terminal 2):
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend will run at `http://localhost:3000`.

3. **Access the application**:
   Open `http://localhost:3000` in your browser

## API Documentation

### POST /generate

Generate a nutrition report from a food image.

**Request**:
- Method: `POST`
- Endpoint: `/generate`
- Content-Type: `multipart/form-data`
- Body: `image` (image file)

**Response**:
```json
{
  "text": "Generated nutrition report...",
  "report": "Generated nutrition report...",
  "filename": "image.jpg",
  "status": "success"
}
```

## Model Information

- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Task**: Food image-to-nutrition analysis
- **Parameters**: Efficient fine-tuning with r=16, lora_alpha=32

## Configuration

### Backend Configuration

Set environment variables for model paths:

- `HF_CACHE_DIR`: Hugging Face cache directory (default: `~/.cache/huggingface`)
- `ADAPTER_PATH`: LoRA adapter path (default: `./output/Qwen2.5-VL-7B-nutrition/checkpoint-215`)

Example:
```bash
export HF_CACHE_DIR=~/.cache/huggingface
export ADAPTER_PATH=./output/Qwen2.5-VL-7B-nutrition/checkpoint-215
```

## Development

### Frontend Development

The frontend uses Vite for fast development. Changes are hot-reloaded automatically.

```bash
cd frontend
npm run dev
```

### Backend Development

The backend uses FastAPI with automatic API documentation available at `http://localhost:8000/docs`.

### Model Training

See `Qwen2.5-VL-Finetune/README.md` for details on model fine-tuning.

## Contributors

- @timo1228
- @Character-Y

## License

[Add your license information here]

## Citation

If you use this project in your research, please cite:

```
NutriVision: A Parameter-Efficient LMM for Food Image-to-Nutrition Analysis
```
