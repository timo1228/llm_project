"""
Test script to verify Gemini API setup with a single dish.

Usage:
    python test_api.py --api-key YOUR_GEMINI_API_KEY
"""

import argparse
from pathlib import Path
import google.generativeai as genai


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


def format_ingredients(ingredients: list) -> str:
    """Format ingredients list for the prompt."""
    sorted_ingrs = sorted(ingredients, key=lambda x: x['grams'], reverse=True)
    formatted = []
    for ing in sorted_ingrs:
        formatted.append(f"{ing['name']} ({ing['grams']:.1f}g)")
    return ', '.join(formatted)


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


def main():
    parser = argparse.ArgumentParser(description='Test Gemini API with a single dish')
    parser.add_argument('--api-key', type=str, required=True, help='Gemini API key')
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    metadata_file = project_dir / 'data' / 'raw' / 'metadata' / 'dish_metadata_cafe1.csv'
    
    # Load first dish
    print("Loading sample dish...")
    with open(metadata_file, 'r') as f:
        first_line = f.readline()
    
    dish = parse_dish_row(first_line)
    print(f"Dish ID: {dish['dish_id']}")
    print(f"Calories: {dish['total_calories']:.1f} kcal")
    print(f"Ingredients: {len(dish['ingredients'])}")
    
    # Format prompt
    ingredients_str = format_ingredients(dish['ingredients'])
    prompt = PROMPT_TEMPLATE.format(
        ingredients=ingredients_str,
        calories=dish['total_calories'],
        mass=dish['total_mass'],
        protein=dish['total_protein'],
        fat=dish['total_fat'],
        carbs=dish['total_carb']
    )
    
    print("\n--- Prompt ---")
    print(prompt[:500] + "...")
    
    # Configure Gemini API
    print("\n--- Testing API ---")
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        response = model.generate_content(prompt)
        print("\n--- Generated Report ---")
        print(response.text)
        print("\n--- API Test Successful! ---")
    except Exception as e:
        print(f"\n--- API Error ---")
        print(f"Error: {e}")
        print("\nPlease check your API key and network connection.")


if __name__ == '__main__':
    main()





