import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info
from PIL import Image
import io

app = FastAPI()

# Global variables for model and processor
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, processor
    
    # Base model path
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache_dir = "D:/cache/huggingface"
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path, 
        cache_dir=cache_dir+"/models",
        trust_remote_code=True
    )
    
    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=cache_dir + "/models"
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
    
    # Load adapter
    adapter_path = "./output/Qwen2.5-VL-7B-nutrition/checkpoint-215"
    model = PeftModel.from_pretrained(base_model, model_id=adapter_path, config=val_config)
    model.eval()
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_model()

def predict(image, prompt="Generate a nutrition report for the given food item:"):
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
        torch.cuda.empty_cache()

        return output_text[0]

@app.post("/generate")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File must be an image"})
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run inference
        result = predict(image)
        
        return {"report": result}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

