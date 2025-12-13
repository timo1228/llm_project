# Nutrition5k Report Generation Pipeline

This directory contains scripts for generating nutritional reports from the Nutrition5k dataset using Gemini 1.5 Flash API.

## Directory Structure

```
Project/
├── data/
│   ├── raw/
│   │   └── metadata/
│   │       ├── dish_metadata_cafe1.csv  (4,768 dishes)
│   │       └── dish_metadata_cafe2.csv  (238 dishes)
│   └── processed/
│       ├── reports.json                  (generated reports)
│       └── image_report_pairs.json       (final training pairs)
├── scripts/
│   ├── verify_data.py        # Verify dataset structure
│   ├── test_api.py           # Test Gemini API connection
│   ├── generate_reports.py   # Main report generation script
│   └── create_pairs.py       # Create image-report pairs
└── requirements.txt
```

## Setup

### 1. Install Dependencies

```bash
pip install google-generativeai tqdm
```

### 2. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the API key for use in the scripts

## Usage

### Step 1: Verify Data (Optional)

```bash
python3 scripts/verify_data.py
```

Expected output:
```
Total dishes loaded: 5006
Total ingredient entries: 28455
Average ingredients per dish: 5.68
```

### Step 2: Test API Connection

```bash
python3 scripts/test_api.py --api-key YOUR_API_KEY
```

This generates a single report to verify your API setup works correctly.

### Step 3: Generate All Reports

```bash
# Generate all 5,006 reports
python3 scripts/generate_reports.py --api-key YOUR_API_KEY

# Or with custom parameters
python3 scripts/generate_reports.py \
    --api-key YOUR_API_KEY \
    --batch-size 100 \
    --delay 0.3 \
    --start-idx 0 \
    --max-dishes 1000  # Optional: limit for testing
```

**Parameters:**
- `--api-key`: Your Gemini API key (required)
- `--batch-size`: Save progress every N dishes (default: 50)
- `--delay`: Delay between API calls in seconds (default: 0.5)
- `--start-idx`: Starting index for processing (default: 0)
- `--max-dishes`: Maximum dishes to process (default: all)

**Progress:** The script saves progress to `data/processed/reports.json` after each batch. If interrupted, it will resume from where it left off.

**Estimated Time:** ~30-60 minutes for 5,006 dishes (with 0.5s delay)

**Estimated Cost:** ~$0.50 with Gemini 1.5 Flash

### Step 4: Create Image-Report Pairs

After generating reports, create the final training dataset:

```bash
# Without images (report-only pairs)
python3 scripts/create_pairs.py

# With images (requires downloaded Nutrition5k imagery)
python3 scripts/create_pairs.py --imagery-dir /path/to/nutrition5k/imagery
```

## Output Format

### reports.json

```json
{
  "dish_1561662216": {
    "report": "Name: Asian Pork Rice Bowl with Mixed Greens\n\nMain Ingredients: ...",
    "metadata": {
      "calories": 300.8,
      "mass": 193.0,
      "protein": 18.6,
      "fat": 12.4,
      "carbs": 28.2,
      "num_ingredients": 17,
      "ingredients": ["soy sauce", "garlic", "white rice", ...]
    }
  },
  ...
}
```

### image_report_pairs.json

```json
[
  {
    "dish_id": "dish_1561662216",
    "report": "Name: Asian Pork Rice Bowl...",
    "images": {
      "overhead_rgb": "/path/to/rgb.png",
      "overhead_depth": "/path/to/depth.png",
      "side_angle_frames": ["/path/to/frame1.jpg", ...]
    },
    "metadata": {...}
  },
  ...
]
```

## Sample Generated Report

```
Name: Asian Pork Rice Bowl with Mixed Greens

Main Ingredients: pork (59.5g), brown rice (68.0g), mixed greens (21.3g), white rice (8.5g), bok choy (8.5g), sugar (6.4g), apple (4.3g), soy sauce (3.4g), olive oil (3.2g), millet (3.4g), garlic (2.1g), onions (1.7g), vinegar (0.9g), lemon juice (0.9g), salt (0.5g), pepper (0.2g), parsley (0.2g)

Nutritional Information (per serving):
- Calories: 300.8 kcal
- Protein: 18.6 g
- Fat: 12.4 g
- Carbohydrates: 28.2 g
- Total Mass: 193.0 g

Nutritional Analysis: This dish provides a well-balanced macronutrient profile with moderate protein content primarily from pork. The combination of brown and white rice provides complex carbohydrates, while the mixed greens and bok choy contribute dietary fiber and micronutrients. The fat content is reasonable, mainly from olive oil and pork, though the added sugar could be reduced for a healthier option.
```

## Downloading Nutrition5k Images

To include images in the training pairs, download the Nutrition5k imagery:

```bash
# Download overhead images only (~10GB)
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" data/raw/imagery/

# Or download side-angle videos (~170GB)
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles" data/raw/imagery/
```

## Troubleshooting

**API Rate Limiting:** If you encounter rate limits, increase the `--delay` parameter.

**Interrupted Generation:** The script automatically saves progress. Simply re-run with the same parameters to resume.

**Missing Packages:** Run `pip install google-generativeai tqdm`

## ROUGE Evaluation

After training your model, you can evaluate its performance using ROUGE metrics:

```bash
# Basic evaluation
python3 scripts/evaluate_rouge.py --checkpoint ./output/Qwen2.5-VL-7B-nutrition/checkpoint-215

# With custom parameters
python3 scripts/evaluate_rouge.py \
    --checkpoint ./output/Qwen2.5-VL-7B-nutrition/checkpoint-215 \
    --test-split 0.2 \
    --max-samples 100 \
    --seed 42 \
    --output-dir results/evaluation \
    --save-details \
    --imagery-dir /path/to/nutrition5k/imagery
```

**Parameters:**
- `--checkpoint`: Path to LoRA adapter checkpoint (required)
- `--test-split`: Test set proportion (default: 0.2)
- `--max-samples`: Maximum number of test samples to evaluate (default: all)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Output directory for results (default: results/evaluation)
- `--save-details`: Save detailed results for each sample
- `--imagery-dir`: Path to Nutrition5k imagery directory (optional)
- `--device`: Device to use: auto, cuda, or cpu (default: auto)

**Output:**
- Console output: ROUGE-1, ROUGE-2, and ROUGE-L scores (precision, recall, F-measure)
- JSON file: `results/evaluation/rouge_evaluation_results.json` with aggregated scores and optional detailed per-sample results

**Dependencies:**
- Install ROUGE evaluation library: `pip install rouge-score` or `pip install datasets[evaluate]`


