"""
Verify and explore the Nutrition5k dataset structure.
"""

from pathlib import Path


def parse_dish_row(row_str: str) -> dict:
    """Parse a single row from the dish metadata CSV."""
    parts = row_str.strip().split(',')
    
    # First 6 fields are dish-level info
    dish_data = {
        'dish_id': parts[0],
        'total_calories': float(parts[1]),
        'total_mass': float(parts[2]),
        'total_fat': float(parts[3]),
        'total_carb': float(parts[4]),
        'total_protein': float(parts[5]),
        'ingredients': []
    }
    
    # Remaining fields are ingredient data (7 fields each)
    # ingr_id, ingr_name, ingr_grams, ingr_calories, ingr_fat, ingr_carb, ingr_protein
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


def main():
    metadata_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'metadata'
    
    print("Loading dishes...")
    dishes = load_all_dishes(metadata_dir)
    print(f"Total dishes loaded: {len(dishes)}")
    
    # Show some statistics
    print("\n--- Dataset Statistics ---")
    total_ingredients = sum(len(d['ingredients']) for d in dishes)
    print(f"Total ingredient entries: {total_ingredients}")
    print(f"Average ingredients per dish: {total_ingredients / len(dishes):.2f}")
    
    # Calorie statistics
    calories = [d['total_calories'] for d in dishes]
    print(f"\nCalories range: {min(calories):.1f} - {max(calories):.1f} kcal")
    print(f"Average calories: {sum(calories) / len(calories):.1f} kcal")
    
    # Mass statistics
    masses = [d['total_mass'] for d in dishes]
    print(f"\nMass range: {min(masses):.1f} - {max(masses):.1f} g")
    print(f"Average mass: {sum(masses) / len(masses):.1f} g")
    
    # Show sample dish
    print("\n--- Sample Dish ---")
    sample = dishes[0]
    print(f"Dish ID: {sample['dish_id']}")
    print(f"Calories: {sample['total_calories']:.1f} kcal")
    print(f"Mass: {sample['total_mass']:.1f} g")
    print(f"Fat: {sample['total_fat']:.1f} g")
    print(f"Carbs: {sample['total_carb']:.1f} g")
    print(f"Protein: {sample['total_protein']:.1f} g")
    print(f"Ingredients ({len(sample['ingredients'])}):")
    for ing in sample['ingredients'][:5]:
        print(f"  - {ing['name']}: {ing['grams']:.1f}g")
    if len(sample['ingredients']) > 5:
        print(f"  ... and {len(sample['ingredients']) - 5} more")


if __name__ == '__main__':
    main()

