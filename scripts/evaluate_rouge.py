"""
ROUGE Evaluation Script for NutriVision Model

This script evaluates the trained Qwen2.5-VL model by comparing generated
nutrition reports against ground truth reports using ROUGE metrics.

Usage:
    python scripts/evaluate_rouge.py --checkpoint ./output/Qwen2.5-VL-7B-nutrition/checkpoint-215
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
    USE_DATASETS = False
except ImportError:
    try:
        import evaluate
        ROUGE_AVAILABLE = True
        USE_DATASETS = True
    except ImportError:
        ROUGE_AVAILABLE = False
        USE_DATASETS = False


def load_reports(reports_file: Path) -> Dict[str, Dict]:
    """Load ground truth reports from JSON file."""
    print(f"Loading reports from {reports_file}")
    with open(reports_file, 'r', encoding='utf-8') as f:
        reports = json.load(f)
    print(f"Loaded {len(reports)} reports")
    return reports


def load_image_pairs(pairs_file: Path) -> Dict[str, Dict]:
    """Load image-report pairs to get image paths."""
    if not pairs_file.exists():
        print(f"Warning: {pairs_file} not found. Will skip samples without explicit image paths.")
        return {}
    
    print(f"Loading image pairs from {pairs_file}")
    with open(pairs_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    # Convert list to dict keyed by dish_id
    pairs_dict = {item['dish_id']: item for item in pairs if 'dish_id' in item}
    print(f"Loaded {len(pairs_dict)} image pairs")
    return pairs_dict


def create_test_split(
    reports: Dict[str, Dict],
    test_split: float = 0.2,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """Create train/test split from reports."""
    dish_ids = list(reports.keys())
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle dish IDs
    shuffled_ids = dish_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Split
    n_test = int(len(shuffled_ids) * test_split)
    test_ids = shuffled_ids[:n_test]
    train_ids = shuffled_ids[n_test:]
    
    # Limit test set size if specified
    if max_samples is not None and len(test_ids) > max_samples:
        test_ids = test_ids[:max_samples]
    
    print(f"Train set: {len(train_ids)} samples")
    print(f"Test set: {len(test_ids)} samples")
    
    return train_ids, test_ids


def load_model(checkpoint_path: str, cache_dir: Optional[str] = None, device: str = "auto"):
    """Load Qwen2.5-VL model with LoRA adapter."""
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    if cache_dir is None:
        cache_dir = os.getenv("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Cache directory: {cache_dir}")
    print("Loading processor...")
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        cache_dir=os.path.join(cache_dir, "models"),
        trust_remote_code=True
    )
    
    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
        cache_dir=os.path.join(cache_dir, "models")
    )
    
    print(f"Loading LoRA adapter from {checkpoint_path}...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    
    model = PeftModel.from_pretrained(base_model, model_id=checkpoint_path, config=lora_config)
    model.eval()
    
    print("Model loaded successfully!")
    
    return model, processor, device


def find_image_path(dish_id: str, image_pairs: Dict[str, Dict], imagery_dir: Optional[Path] = None) -> Optional[str]:
    """Find image path for a given dish_id."""
    # First, try to find in image_pairs
    if dish_id in image_pairs:
        pair = image_pairs[dish_id]
        if 'images' in pair and pair['images']:
            # Prefer overhead RGB image
            if 'overhead_rgb' in pair['images'] and pair['images']['overhead_rgb']:
                img_path = pair['images']['overhead_rgb']
                if os.path.exists(img_path):
                    return img_path
            # Fallback to first available image
            if isinstance(pair['images'], dict):
                for key in ['overhead_depth', 'side_angle_frames']:
                    if key in pair['images'] and pair['images'][key]:
                        if key == 'side_angle_frames' and isinstance(pair['images'][key], list):
                            if len(pair['images'][key]) > 0 and os.path.exists(pair['images'][key][0]):
                                return pair['images'][key][0]
                        elif os.path.exists(pair['images'][key]):
                            return pair['images'][key]
    
    # If imagery_dir is provided, try to find image there
    if imagery_dir:
        overhead_rgb = imagery_dir / 'realsense_overhead' / dish_id / 'rgb.png'
        if overhead_rgb.exists():
            return str(overhead_rgb)
    
    return None


def predict(model, processor, device, image_path: str, prompt: str = "Generate a nutrition report for the given food item:"):
    """Generate nutrition report for an image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    with torch.no_grad():
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
        
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        del inputs
        del generated_ids
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return output_text[0]


def compute_rouge_scores(predictions: List[str], references: List[str], use_datasets: bool = False):
    """Compute ROUGE scores between predictions and references."""
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge-score or datasets library is required. Install with: pip install rouge-score or pip install datasets[evaluate]")
    
    if use_datasets:
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=predictions, references=references)
        # Convert to standard format
        return {
            'rouge1': {
                'precision': results.get('rouge1', 0),
                'recall': results.get('rouge1', 0),
                'fmeasure': results.get('rouge1', 0)
            },
            'rouge2': {
                'precision': results.get('rouge2', 0),
                'recall': results.get('rouge2', 0),
                'fmeasure': results.get('rouge2', 0)
            },
            'rougel': {
                'precision': results.get('rougeL', 0),
                'recall': results.get('rougeL', 0),
                'fmeasure': results.get('rougeL', 0)
            }
        }
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
        
        # Aggregate scores
        aggregated = {'rouge1': [], 'rouge2': [], 'rougel': []}
        for score in scores:
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                if metric == 'rougeL':
                    key = 'rougel'
                else:
                    key = metric
                aggregated[key].append({
                    'precision': score[metric].precision,
                    'recall': score[metric].recall,
                    'fmeasure': score[metric].fmeasure
                })
        
        # Average scores
        result = {}
        for metric in ['rouge1', 'rouge2', 'rougel']:
            precisions = [s['precision'] for s in aggregated[metric]]
            recalls = [s['recall'] for s in aggregated[metric]]
            fmeasures = [s['fmeasure'] for s in aggregated[metric]]
            
            result[metric] = {
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'fmeasure': np.mean(fmeasures)
            }
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate model using ROUGE metrics')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to LoRA adapter checkpoint'
    )
    parser.add_argument(
        '--reports-file',
        type=str,
        default=None,
        help='Path to reports.json (default: data/processed/reports.json)'
    )
    parser.add_argument(
        '--pairs-file',
        type=str,
        default=None,
        help='Path to image_report_pairs.json (default: data/image_report_pairs.json)'
    )
    parser.add_argument(
        '--imagery-dir',
        type=str,
        default=None,
        help='Path to Nutrition5k imagery directory (optional)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of test samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results (default: results/evaluation)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='HuggingFace cache directory (default: ~/.cache/huggingface)'
    )
    parser.add_argument(
        '--save-details',
        action='store_true',
        help='Save detailed results for each sample'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    reports_file = Path(args.reports_file) if args.reports_file else project_dir / 'data' / 'processed' / 'reports.json'
    pairs_file = Path(args.pairs_file) if args.pairs_file else project_dir / 'data' / 'image_report_pairs.json'
    imagery_dir = Path(args.imagery_dir) if args.imagery_dir else None
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    reports = load_reports(reports_file)
    image_pairs = load_image_pairs(pairs_file)
    
    # Create test split
    train_ids, test_ids = create_test_split(reports, args.test_split, args.seed, args.max_samples)
    
    # Load model
    model, processor, device = load_model(args.checkpoint, args.cache_dir, args.device)
    
    # Evaluate on test set
    print(f"\nEvaluating on {len(test_ids)} test samples...")
    predictions = []
    references = []
    detailed_results = []
    skipped = 0
    
    for dish_id in tqdm(test_ids, desc="Evaluating"):
        if dish_id not in reports:
            skipped += 1
            continue
        
        ground_truth = reports[dish_id]['report']
        
        # Find image path
        image_path = find_image_path(dish_id, image_pairs, imagery_dir)
        
        if image_path is None:
            skipped += 1
            if args.save_details:
                detailed_results.append({
                    'dish_id': dish_id,
                    'ground_truth': ground_truth,
                    'prediction': None,
                    'error': 'Image not found',
                    'rouge_scores': None
                })
            continue
        
        try:
            # Generate prediction
            prediction = predict(model, processor, device, image_path)
            
            predictions.append(prediction)
            references.append(ground_truth)
            
            if args.save_details:
                # Compute individual ROUGE scores
                try:
                    try:
                        import evaluate
                        rouge = evaluate.load('rouge')
                        individual_scores = rouge.compute(predictions=[prediction], references=[ground_truth])
                        rouge_dict = {
                            'rouge1': individual_scores.get('rouge1', 0),
                            'rouge2': individual_scores.get('rouge2', 0),
                            'rougel': individual_scores.get('rougeL', 0)
                        }
                    except ImportError:
                        from rouge_score import rouge_scorer
                        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                        score = scorer.score(ground_truth, prediction)
                        rouge_dict = {
                            'rouge1': score['rouge1'].fmeasure,
                            'rouge2': score['rouge2'].fmeasure,
                            'rougel': score['rougeL'].fmeasure
                        }
                except Exception as e:
                    print(f"Warning: Failed to compute individual ROUGE for {dish_id}: {e}")
                    rouge_dict = None
                
                detailed_results.append({
                    'dish_id': dish_id,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'image_path': image_path,
                    'rouge_scores': rouge_dict
                })
        
        except Exception as e:
            skipped += 1
            print(f"Error processing {dish_id}: {e}")
            if args.save_details:
                detailed_results.append({
                    'dish_id': dish_id,
                    'ground_truth': ground_truth,
                    'prediction': None,
                    'error': str(e),
                    'rouge_scores': None
                })
    
    print(f"\nEvaluation complete!")
    print(f"Successfully evaluated: {len(predictions)} samples")
    print(f"Skipped: {skipped} samples")
    
    if len(predictions) == 0:
        print("Error: No predictions generated. Check image paths and model.")
        return
    
    # Compute overall ROUGE scores
    print("\nComputing ROUGE scores...")
    rouge_scores = compute_rouge_scores(predictions, references, use_datasets=USE_DATASETS)
    
    # Print results
    print("\n" + "="*60)
    print("ROUGE Evaluation Results")
    print("="*60)
    for metric in ['rouge1', 'rouge2', 'rougel']:
        scores = rouge_scores[metric]
        print(f"\n{metric.upper()}:")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F-measure: {scores['fmeasure']:.4f}")
    
    # Save results
    results = {
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougel': rouge_scores['rougel'],
        'num_samples': len(predictions),
        'num_skipped': skipped,
        'checkpoint': args.checkpoint,
        'test_split': args.test_split,
        'seed': args.seed
    }
    
    if args.save_details:
        results['detailed_results'] = detailed_results
    
    output_file = output_dir / 'rouge_evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    print("="*60)


if __name__ == '__main__':
    main()

