"""
Generate nutritional reports for Nutrition5k dishes using Gemini 1.5 Flash API.

Usage:
    python generate_reports.py --api-key YOUR_GEMINI_API_KEY [--batch-size 50] [--start-idx 0]

Requirements:
    pip install google-generativeai tqdm
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from tqdm import tqdm


# Prompt template for generating nutritional reports
PROMPT_TEMPLATE = """You are a nutritionist assistant. Given the following nutritional data for a food dish, generate a detailed nutritional report in English.

Dish Data:
- Ingredients: {ingredients}
- Total Calories: {calories:.1f} kcal
- Total Mass: {mass:.1f} g
- Protein: {protein:.1f} g
- Fat: {fat:.1f} g
- Carbohydrates: {carbs:.1f} g

Generate a report with the following exact format (no additional text or explanation):

Name: [Create a descriptive dish name based on the ingredients, e.g., "Grilled Chicken Salad with Mixed Greens"]

Main Ingredients: [List main ingredients with their weights in format: ingredient1 (Xg), ingredient2 (Yg), ...]

Nutritional Information (per serving):
- Calories: {calories:.1f} kcal
- Protein: {protein:.1f} g
- Fat: {fat:.1f} g
- Carbohydrates: {carbs:.1f} g
- Total Mass: {mass:.1f} g

Nutritional Analysis: [Write 2-3 sentences analyzing the nutritional profile. Consider protein-to-calorie ratio, fat content, fiber sources, and provide brief dietary recommendations or health considerations.]

Important: Keep the analysis factual, concise, and professional. Do not use emojis or informal language."""


def parse_dish_row(row_str: str) -> dict:
    """Parse a single row from the dish metadata CSV."""
    parts = row_str.strip().split(',')
    
    dish_data = {
        'dish_id': parts[0],
        'total_calories': float(parts[1]),
        'total_mass': float(parts[2]),
        'total_fat': float(parts[3]),
        'total_carb': float(parts[4]),
        'total_protein': float(parts[5]),
        'ingredients': []
    }
    
    idx = 6
    while idx + 6 < len(parts):
        ingredient = {
            'id': parts[idx],
            'name': parts[idx + 1],
            'grams': float(parts[idx + 2]),
            'calories': float(parts[idx + 3]),
            'fat': float(parts[idx + 4]),
            'carb': float(parts[idx + 5]),
            'protein': float(parts[idx + 6])
        }
        dish_data['ingredients'].append(ingredient)
        idx += 7
    
    return dish_data


def load_all_dishes(metadata_dir: Path) -> list:
    """Load all dishes from both cafe metadata files."""
    all_dishes = []
    
    for csv_file in ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']:
        filepath = metadata_dir / csv_file
        if filepath.exists():
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        dish = parse_dish_row(line)
                        all_dishes.append(dish)
    
    return all_dishes


def format_ingredients(ingredients: list) -> str:
    """Format ingredients list for the prompt."""
    # Sort by weight (descending) and take top ingredients
    sorted_ingrs = sorted(ingredients, key=lambda x: x['grams'], reverse=True)
    formatted = []
    for ing in sorted_ingrs:
        formatted.append(f"{ing['name']} ({ing['grams']:.1f}g)")
    return ', '.join(formatted)


def create_prompt(dish: dict) -> str:
    """Create a prompt for generating a nutritional report."""
    ingredients_str = format_ingredients(dish['ingredients'])
    
    return PROMPT_TEMPLATE.format(
        ingredients=ingredients_str,
        calories=dish['total_calories'],
        mass=dish['total_mass'],
        protein=dish['total_protein'],
        fat=dish['total_fat'],
        carbs=dish['total_carb']
    )


def generate_report(model, dish: dict, max_retries: int = 3) -> Optional[str]:
    """Generate a nutritional report for a single dish using Gemini API."""
    prompt = create_prompt(dish)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"\nError for {dish['dish_id']}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed to generate report for {dish['dish_id']} after {max_retries} attempts: {e}")
                return None
    
    return None


def load_progress(output_file: Path) -> dict:
    """Load existing progress from output file."""
    if output_file.exists():
        with open(output_file, 'r') as f:
            return json.load(f)
    return {}


def save_progress(reports: dict, output_file: Path):
    """Save progress to output file."""
    with open(output_file, 'w') as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Generate nutritional reports using Gemini API')
    parser.add_argument('--api-key', type=str, required=True, help='Gemini API key')
    parser.add_argument('--batch-size', type=int, default=50, help='Number of dishes to process before saving')
    parser.add_argument('--start-idx', type=int, default=0, help='Starting index for processing')
    parser.add_argument('--max-dishes', type=int, default=None, help='Maximum number of dishes to process')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls in seconds')
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    metadata_dir = project_dir / 'data' / 'raw' / 'metadata'
    output_dir = project_dir / 'data' / 'processed'
    output_file = output_dir / 'reports.json'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure Gemini API
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Load dishes
    print("Loading dishes...")
    dishes = load_all_dishes(metadata_dir)
    print(f"Total dishes: {len(dishes)}")
    
    # Load existing progress
    reports = load_progress(output_file)
    print(f"Existing reports: {len(reports)}")
    
    # Determine which dishes to process
    if args.max_dishes:
        dishes = dishes[args.start_idx:args.start_idx + args.max_dishes]
    else:
        dishes = dishes[args.start_idx:]
    
    # Filter out already processed dishes
    dishes_to_process = [d for d in dishes if d['dish_id'] not in reports]
    print(f"Dishes to process: {len(dishes_to_process)}")
    
    if not dishes_to_process:
        print("All dishes already processed!")
        return
    
    # Process dishes
    processed_count = 0
    
    for dish in tqdm(dishes_to_process, desc="Generating reports"):
        report = generate_report(model, dish)
        
        if report:
            reports[dish['dish_id']] = {
                'report': report,
                'metadata': {
                    'calories': dish['total_calories'],
                    'mass': dish['total_mass'],
                    'protein': dish['total_protein'],
                    'fat': dish['total_fat'],
                    'carbs': dish['total_carb'],
                    'num_ingredients': len(dish['ingredients']),
                    'ingredients': [ing['name'] for ing in dish['ingredients']]
                }
            }
            processed_count += 1
        
        # Save progress periodically
        if processed_count % args.batch_size == 0:
            save_progress(reports, output_file)
            print(f"\nSaved progress: {len(reports)} reports")
        
        # Rate limiting
        time.sleep(args.delay)
    
    # Final save
    save_progress(reports, output_file)
    print(f"\nCompleted! Total reports: {len(reports)}")
    print(f"Output saved to: {output_file}")


if __name__ == '__main__':
    main()


