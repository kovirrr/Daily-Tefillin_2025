import os
import shutil
import random

def organize_data(base_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Create necessary directories
    for split in ['train', 'val', 'test']:
        for data_type in ['images', 'labels']:
            os.makedirs(os.path.join(base_dir, data_type, split), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(base_dir, 'images')) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    
    # Calculate split indices
    total_images = len(image_files)
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * (train_ratio + val_ratio))
    
    # Split and move files
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(base_dir, 'images', img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(base_dir, 'labels', label_file)
        
        if i < train_split:
            split = 'train'
        elif i < val_split:
            split = 'val'
        else:
            split = 'test'
        
        # Move image
        shutil.move(img_path, os.path.join(base_dir, 'images', split, img_file))
        
        # Move label if it exists
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(base_dir, 'labels', split, label_file))
        else:
            print(f"Warning: Label file not found for {img_file}")

    print("Data organization completed.")

# Usage
base_dir = '/Users/koviressler/Desktop/DailyTefillin/hairline_detection'
organize_data(base_dir)