"""
NutriVision Backend API
A Parameter-Efficient LMM for Food Image-to-Nutrition Analysis

FastAPI backend server for image upload and nutrition report generation.
Uses Qwen2.5-VL-7B-Instruct model with LoRA adapter.
Endpoint: POST /generate
"""

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import os

app = FastAPI(title="NutriVision API", description="A Parameter-Efficient LMM for Food Image-to-Nutrition Analysis")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load the Qwen2.5-VL model with LoRA adapter."""
    global model, processor
    
    # Base model path
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Cache directory - can be set via environment variable or use default
    cache_dir = os.getenv("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))
    
    print(f"Using device: {device}")
    print(f"Cache directory: {cache_dir}")
    print("Loading processor...")
    
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, 
            cache_dir=os.path.join(cache_dir, "models"),
            trust_remote_code=True
        )
        
        print("Loading base model...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=os.path.join(cache_dir, "models")
        )
        
        print("Loading LoRA adapter...")
        # LoRA config
        val_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
        
        # Load adapter - adjust path as needed
        adapter_path = os.getenv("ADAPTER_PATH", "./output/Qwen2.5-VL-7B-nutrition/checkpoint-215")
        model = PeftModel.from_pretrained(base_model, model_id=adapter_path, config=val_config)
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Server will start but /generate endpoint will not work until model is loaded.")


def predict(image, prompt="Generate a nutrition report for the given food item:"):
    """
    Generate nutrition report for the given image.
    
    Args:
        image: PIL Image object
        prompt: Text prompt for the model
    
    Returns:
        Generated text report
    """
    if model is None or processor is None:
        raise RuntimeError("Model not loaded. Please check server logs.")
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": prompt,
            }
        ]
    }]

    with torch.no_grad():
        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Cleanup
        del inputs
        del generated_ids
        if device == "cuda":
            torch.cuda.empty_cache()

        return output_text[0]


@app.get("/")
async def root():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "message": "NutriVision API is running",
        "project": "NutriVision: A Parameter-Efficient LMM for Food Image-to-Nutrition Analysis",
        "endpoint": "/generate",
        "model_status": model_status,
        "device": device
    }


@app.post("/generate")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Receive an image file and return generated nutrition report.
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON response with generated report text
    """
    if model is None or processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs and ensure model files are available."
        )
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        result = predict(image)
        
        # Return response in format compatible with frontend
        # Frontend expects: data.text || data.result || JSON.stringify(data)
        return JSONResponse(content={
            "text": result,  # Primary field for frontend
            "report": result,  # Also include 'report' field for compatibility
            "filename": file.filename,
            "status": "success"
        })
        
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model inference error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图片时出错: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
