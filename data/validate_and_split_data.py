import json
import os

if __name__ == "__main__":
    # Define paths (relative to the current script location)
    # 假設腳本在 data/ 目錄下運行
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'image_report_pairs.json')
    images_dir = os.path.join(base_dir, 'nutrition5k_dataset/imagery/realsense_overhead/')
    output_train_path = os.path.join(base_dir, 'nutrition5k_dataset/data_vl_train.json')
    output_eval_path = os.path.join(base_dir, 'nutrition5k_dataset/data_vl_eval.json')

    # Load the JSON data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {json_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        data = []

    # Check for existence of dish_id directories and format data
    formatted_items = []
    missing_dishes = []

    print("Checking for image existence and formatting...")
    for idx, item in enumerate(data):
        dish_id = item['dish_id']
        # 假設圖片名為 rgb.png，這是 nutrition5k 的常見結構
        image_rel_path = os.path.join(images_dir, dish_id, 'rgb.png')
        
        # 檢查文件是否存在
        if os.path.isfile(image_rel_path):
            # 獲取絕對路徑，確保訓練時能找到
            abs_image_path = os.path.abspath(image_rel_path)
            
            # 構建新的數據格式
            new_item = {
                "id": f"identity_{idx}",
                "conversations": [
                    {
                        "from": "user",
                        "value": f"<|vision_start|>{abs_image_path}<|vision_end|>Generate a nutrition report for the given food item:"
                    },
                    {
                        "from": "assistant",
                        "value": item['report']
                    }
                ]
            }
            formatted_items.append(new_item)
        else:
            missing_dishes.append(dish_id)

    # Report results
    print(f"Total dishes in JSON: {len(data)}")
    print(f"Valid items (found in imagery): {len(formatted_items)}")
    print(f"Missing items: {len(missing_dishes)}")

    # Split into train and eval
    if len(formatted_items) > 50:
        eval_data = formatted_items[-50:]
        train_data = formatted_items[:-50]

        print(f"Split results:")
        print(f"Train size: {len(train_data)}")
        print(f"Eval size: {len(eval_data)}")

        # Save to files
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_train_path), exist_ok=True)

        with open(output_train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        print(f"Saved train data to {output_train_path}")

        with open(output_eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        print(f"Saved eval data to {output_eval_path}")
    else:
        print("Not enough valid items to split (need > 50).")
