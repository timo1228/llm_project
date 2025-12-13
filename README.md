# NutriVision: Advanced Food Nutrition Analysis with Multimodal LLMs

NutriVision is an advanced AI system designed to analyze food images and generate detailed nutritional reports. This project explores and implements various Vision-Language Models (VLMs), specifically focusing on **Qwen2.5-VL-7B** (fine-tuned) and **BLIP-2**, to achieve accurate food recognition and ingredient analysis.

The system includes a complete pipeline from data processing and model fine-tuning to a user-friendly Web Demo.

---

## 🏗️ Project Architecture

This repository contains two main modules:

1.  **Qwen2.5-VL Finetuning (`Qwen2.5-VL-Finetune/`)**:
    *   Based on the state-of-the-art **Qwen2.5-VL-7B-Instruct** model.
    *   Implements **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.
    *   Designed for the **Nutrition5k** dataset to learn detailed food descriptions and nutritional facts.
    *   Includes a FastAPI backend and React frontend for demonstration.

2.  **BLIP-2 Exploration (`blip2/`)**:
    *   Based on **LAVIS (Salesforce)** library.
    *   Exploration of BLIP-2 architecture for image captioning and VQA tasks.
    *   Includes custom task implementations and evaluation scripts.

---

## 🚀 Quick Start

### 1. Environment Setup

It is recommended to use Conda to manage environments.

**For Qwen2.5-VL:**
```bash
conda create -n nutrivision python=3.10
conda activate nutrivision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version
pip install transformers peft accelerate datasets bitsandbytes
pip install fastapi uvicorn python-multipart
```

**For BLIP-2:**
Refer to the `requirements.txt` inside `blip2/LAVIS/` or install `salesforce-lavis`.

### 2. Data Preparation

This project uses the **Nutrition5k** dataset.

1.  Download the dataset (Imagery and Metadata) into `data/nutrition5k_dataset/`.
2.  Preprocess and split the data:
    ```bash
    python data/validate_and_split_data.py
    ```
    This script will:
    *   Validate image paths.
    *   Generate `data_vl_train.json` and `data_vl_eval.json` in Qwen-VL format.

### 3. Fine-tuning Qwen2.5-VL

To fine-tune the model on your custom dataset:

```bash
cd Qwen2.5-VL-Finetune
python train.py \
    --pretrained_model "Qwen/Qwen2.5-VL-7B-Instruct" \
    --train_dataset_path "../data/nutrition5k_dataset/data_vl_train.json" \
    --output_dir "./output/Qwen2.5-VL-7B-nutrition" \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4
```

*Note: The script supports Gradient Checkpointing to save VRAM.*

### 4. Evaluation & Inference

You can evaluate the trained model using the provided Jupyter Notebook:
*   Open `Qwen2.5-VL-Finetune/eval.ipynb`.
*   Load the base model and your trained LoRA adapter.
*   Run inference on test images to generate nutrition reports.

### 5. Web Demo (Full Stack)

This project provides a complete Web UI for testing.

**Start the Backend (FastAPI):**
```bash
cd Qwen2.5-VL-Finetune/frontend/backend
python main.py
```
*   Server runs at: `http://0.0.0.0:8000`
*   Swagger UI: `http://localhost:8000/docs`

**Start the Frontend (React):**
```bash
cd Qwen2.5-VL-Finetune/frontend/frontend
npm install
npm run dev
```
*   Access the UI at: `http://localhost:5173` (or provided local URL)

---

## 📂 Directory Structure

```
.
├── Qwen2.5-VL-Finetune/    # Main module for Qwen-VL
│   ├── train.py            # Fine-tuning script
│   ├── eval.ipynb          # Evaluation notebook
│   ├── web.py              # Simple backend script
│   ├── frontend/           # Full-stack Web Demo source code
│   │   ├── backend/        # FastAPI server
│   │   └── frontend/       # React application
│   └── output/             # Model checkpoints
├── blip2/                  # BLIP-2 exploration module
│   ├── custom/             # Custom model/task definitions
│   └── pretrain_eval.ipynb # Pre-training evaluation
├── data/                   # Dataset directory
│   ├── nutrition5k_dataset/
│   └── validate_and_split_data.py
└── README.md               # Project documentation
```

## 📝 License

[MIT License](LICENSE) (or specify your license)

## 🤝 Acknowledgements

*   [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
*   [LAVIS (Salesforce)](https://github.com/salesforce/LAVIS)
*   [Nutrition5k Dataset](https://github.com/google-research-datasets/nutrition5k)

