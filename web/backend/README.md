# Nutrition Report Generation Backend Service

API for generating nutrition reports based on the Qwen2.5-VL-7B-Instruct model and LoRA adapter.

## Requirements

- Python 3.10+
- CUDA (Optional, for GPU acceleration)
- Sufficient disk space for model files (approx. 15GB)

## Installation

```bash
conda activate image-upload
cd backend
pip install -r requirements.txt
```

## Configuration

### Environment Variables

You can configure model paths via environment variables:

- `HF_CACHE_DIR`: Hugging Face model cache directory (Default: `~/.cache/huggingface`)
- `ADAPTER_PATH`: LoRA adapter path (Default: `./output/Qwen2.5-VL-7B-nutrition/checkpoint-215`)

Example (Windows):
```bash
set HF_CACHE_DIR=D:/cache/huggingface
set ADAPTER_PATH=./output/Qwen2.5-VL-7B-nutrition/checkpoint-215
```

Example (Linux/Mac):
```bash
export HF_CACHE_DIR=~/.cache/huggingface
export ADAPTER_PATH=./output/Qwen2.5-VL-7B-nutrition/checkpoint-215
```

### Model Files

Ensure the following files/directories exist:

1. **Base Model**: Will be automatically downloaded from Hugging Face to `HF_CACHE_DIR/models/Qwen/Qwen2.5-VL-7B-Instruct/`
2. **LoRA Adapter**: Should be located at the path specified by `ADAPTER_PATH`

## Running the Service

```bash
conda activate image-upload
cd backend
python main.py
```

The service will start at `http://localhost:8000`.

## API Endpoints

### GET /

Health check endpoint, returns service status.

**Response Example**:
```json
{
  "message": "Nutrition Report Generation API is running",
  "endpoint": "/generate",
  "model_status": "loaded",
  "device": "cuda"
}
```

### POST /generate

Receives an image file and returns the generated nutrition report.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (Image file)

**Response**:
```json
{
  "text": "Generated nutrition report text...",
  "report": "Generated nutrition report text...",
  "filename": "image.jpg",
  "status": "success"
}
```

**Error Responses**:
- `400`: File is not an image
- `500`: Error processing image
- `503`: Model not loaded or inference error

## Notes

1. **First Run**: The first run will download model files from Hugging Face, which may take a significant amount of time.
2. **Memory Requirements**: The model requires substantial RAM/VRAM, recommended at least 16GB RAM or 8GB VRAM.
3. **GPU Acceleration**: If the system has CUDA, GPU acceleration will be used automatically.
4. **Model Loading**: The model loads when the service starts, which may take a few minutes.

## Troubleshooting

### Model Load Failure

- Check if `HF_CACHE_DIR` path is correct.
- Ensure there is sufficient disk space.
- Check network connection (required for downloading the model).

### Adapter Load Failure

- Check if `ADAPTER_PATH` path is correct.
- Ensure adapter files are complete.

### CUDA Errors

- If using CPU, ensure `device` is set to `"cpu"`.
- Check if PyTorch CUDA version matches the system CUDA version.

### Out of Memory

- Reduce the `max_new_tokens` parameter.
- Use CPU mode (though it will be slower).
- Consider using a quantized model.
