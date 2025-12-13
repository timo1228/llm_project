"""
Create image-report pairs for model fine-tuning from generated reports and Nutrition5k images.

This script maps generated reports to their corresponding images from the Nutrition5k dataset.

Usage:
    python create_pairs.py --imagery-dir /path/to/nutrition5k/imagery

Requirements:
    - Generated reports in data/processed/reports.json
    - Nutrition5k imagery data (side_angles or realsense_overhead)
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def find_image_paths(dish_id: str, imagery_dir: Path) -> dict:
    """
    Find available image paths for a given dish.
    
    Returns dict with:
        - overhead_rgb: Path to overhead RGB image (if available)
        - overhead_depth: Path to overhead depth image (if available)
        - side_angle_frames: List of paths to side-angle video frames (if available)
    """
    result = {
        'overhead_rgb': None,
        'overhead_depth': None,
        'side_angle_frames': []
    }
    
    # Check for overhead images
    overhead_dir = imagery_dir / 'realsense_overhead' / dish_id
    if overhead_dir.exists():
        rgb_path = overhead_dir / 'rgb.png'
        depth_path = overhead_dir / 'depth_color.png'
        
        if rgb_path.exists():
            result['overhead_rgb'] = str(rgb_path)
        if depth_path.exists():
            result['overhead_depth'] = str(depth_path)
    
    # Check for side-angle video frames
    side_angle_dir = imagery_dir / 'side_angles' / dish_id
    if side_angle_dir.exists():
        frames_dir = side_angle_dir / 'frames'
        if frames_dir.exists():
            # Get frames from camera A (or any available camera)
            for cam in ['camera_A', 'camera_B', 'camera_C', 'camera_D']:
                cam_frames = list(frames_dir.glob(f'{cam}_frame_*.jpg'))
                if cam_frames:
                    # Sample every 5th frame as mentioned in the paper
                    cam_frames = sorted(cam_frames)
                    sampled = cam_frames[::5][:10]  # Take up to 10 frames
                    result['side_angle_frames'].extend([str(f) for f in sampled])
    
    return result


def create_training_pairs(
    reports_file: Path,
    imagery_dir: Optional[Path],
    output_file: Path,
    include_metadata: bool = True
):
    """
    Create image-report pairs for training.
    
    Output format (JSON):
    [
        {
            "dish_id": "dish_xxxxx",
            "report": "Name: ...\n\nMain Ingredients: ...",
            "images": {
                "overhead_rgb": "/path/to/rgb.png",
                "overhead_depth": "/path/to/depth.png",
                "side_angle_frames": ["/path/to/frame1.jpg", ...]
            },
            "metadata": {
                "calories": 300.0,
                "mass": 200.0,
                ...
            }
        },
        ...
    ]
    """
    # Load reports
    print(f"Loading reports from {reports_file}")
    with open(reports_file, 'r') as f:
        reports = json.load(f)
    
    print(f"Total reports: {len(reports)}")
    
    training_pairs = []
    missing_images = 0
    
    for dish_id, data in reports.items():
        pair = {
            'dish_id': dish_id,
            'report': data['report'],
            'images': None
        }
        
        if include_metadata and 'metadata' in data:
            pair['metadata'] = data['metadata']
        
        # Find images if imagery directory is provided
        if imagery_dir and imagery_dir.exists():
            image_paths = find_image_paths(dish_id, imagery_dir)
            
            # Check if any images were found
            has_images = (
                image_paths['overhead_rgb'] or 
                image_paths['overhead_depth'] or 
                image_paths['side_angle_frames']
            )
            
            if has_images:
                pair['images'] = image_paths
            else:
                missing_images += 1
        
        training_pairs.append(pair)
    
    # Save output
    print(f"Saving {len(training_pairs)} pairs to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print("\n--- Statistics ---")
    print(f"Total pairs: {len(training_pairs)}")
    
    if imagery_dir:
        with_images = sum(1 for p in training_pairs if p['images'])
        print(f"Pairs with images: {with_images}")
        print(f"Pairs without images: {missing_images}")
    else:
        print("No imagery directory provided - image paths not populated")
    
    return training_pairs


def main():
    parser = argparse.ArgumentParser(description='Create image-report pairs for training')
    parser.add_argument(
        '--imagery-dir', 
        type=str, 
        default=None,
        help='Path to Nutrition5k imagery directory (optional)'
    )
    parser.add_argument(
        '--reports-file',
        type=str,
        default=None,
        help='Path to reports.json (default: data/processed/reports.json)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: data/processed/image_report_pairs.json)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Exclude metadata from output'
    )
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    reports_file = Path(args.reports_file) if args.reports_file else project_dir / 'data' / 'processed' / 'reports.json'
    output_file = Path(args.output_file) if args.output_file else project_dir / 'data' / 'processed' / 'image_report_pairs.json'
    imagery_dir = Path(args.imagery_dir) if args.imagery_dir else None
    
    if not reports_file.exists():
        print(f"Error: Reports file not found: {reports_file}")
        print("Please run generate_reports.py first to generate reports.")
        return
    
    create_training_pairs(
        reports_file=reports_file,
        imagery_dir=imagery_dir,
        output_file=output_file,
        include_metadata=not args.no_metadata
    )
    
    print(f"\nOutput saved to: {output_file}")


if __name__ == '__main__':
    main()


