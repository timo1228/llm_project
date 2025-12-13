# Qwen2.5-VL-Finetune

## 1. Environment Setup

1. Ensure that your computer has at least one NVIDIA GPU and the CUDA environment is properly installed.
2. Install Python (version >= 3.9) and PyTorch with CUDA support.
3. Install the third-party libraries related to Qwen2.5-VL fine-tuning using the following command:

```bash
pip install modelscope transformers sentencepiece accelerate datasets peft swanlab qwen-vl-utils pandas
```

If you encounter errors with modelscope, try running:
```bash
pip install modelscope[framework]
```

## 2. Prepare Data

You need to download the nutrition5k from the github: https://github.com/google-research-datasets/Nutrition5k

download the dataset under /data/nutrition5k_dataset

Then run /data/validate_and_split_data.py to get our fine-tuning dataset json files. Though I already upload this files in github, but the image path is absolute location of my computer , you have to regenerate these files.

## 3. Train

**Single GPU:**

Just run ./train.py, for more specific configuration, see the training script for details, it's not complicated.

**Multi-GPU:**

not supported yet

## 4. Evaluation

See `./eval.ipynb`.
