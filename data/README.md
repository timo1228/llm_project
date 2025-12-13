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
│   ├── src/
│   │   ├── App.jsx       # Main application component
│   │   ├── main.jsx      # Entry point
│   │   └── index.css     # Global styles
│   ├── package.json
│   └── vite.config.js
├── backend/              # FastAPI backend server
│   ├── main.py          # API server with model inference
│   ├── requirements.txt # Python dependencies
│   └── README.md        # Backend documentation
├── Qwen2.5-VL-Finetune/ # Model fine-tuning code and scripts
│   ├── train.py         # Training script
│   ├── eval.ipynb       # Evaluation notebook
│   └── qwen-vl-finetune/ # Fine-tuning utilities
├── blip2/                # BLIP-2 model related code
├── data/                 # Data files (raw and processed)
│   ├── raw/             # Raw data
│   └── processed/       # Processed data
├── scripts/              # Utility scripts
│   ├── generate_reports.py
│   └── test_api.py
├── reports.json          # Generated nutrition reports
├── setup_conda_env.sh   # Environment setup script
└── README.md            # This file
```

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
   cd backend
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**:
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. **Start the backend server** (Terminal 1):
   ```bash
   conda activate image-upload
   cd backend
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
