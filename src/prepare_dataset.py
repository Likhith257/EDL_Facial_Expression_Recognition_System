import os
import cv2
import numpy as np
from tqdm import tqdm

def load_dataset(data_dir="C:\EDL\EDL_Proj2\deepface\DATASET - To Students", target_size=(224, 224), valid_extensions=None, label_depth=2):
    """
    Traverse nested folders to load images and labels, designed for deeper structures.
    Args:
        data_dir (str): Path to dataset folder.
        target_size (tuple): Resize images to (height, width).
        valid_extensions (tuple): Image extensions to include.
        label_depth (int): Depth for label extraction (e.g., 2 for grandparent folder).
    Returns:
        images (np.array): Preprocessed images.
        labels (np.array): Corresponding emotion labels.
    """
    if valid_extensions is None:
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.jfif')  # Broad support
    
    images = []
    labels = []
    total_folders = 0
    total_files = 0
    loaded_files = 0
    skipped_files = 0
    
    print(f"Scanning dataset in: {data_dir}")
    for root, dirs, files in tqdm(os.walk(data_dir), desc="Loading dataset"):
        total_folders += 1
        total_files += len(files)
        
        if files:
            relative_path = root.replace(data_dir, '').strip(os.sep).split(os.sep)
            if len(relative_path) >= label_depth:
                label = relative_path[-label_depth]  # Extract label from specified depth
                print(f"Folder '{os.path.basename(root)}' (label: {label}, {len(files)} files): {files[:3]}...")
            else:
                print(f"Folder '{os.path.basename(root)}' (no valid label, {len(files)} files): {files[:3]}...")
        
        for file in files:
            if file.lower().endswith(valid_extensions):
                img_path = os.path.join(root, file)
                path_parts = root.replace(data_dir, '').strip(os.sep).split(os.sep)
                if len(path_parts) >= label_depth:
                    label = path_parts[-label_depth]
                else:
                    label = 'unknown'  # Fallback if depth exceeded
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load (corrupted?): {img_path}")
                    skipped_files += 1
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                img = img / 255.0
                images.append(img)
                labels.append(label)
                loaded_files += 1
    
    print(f"\nDataset Summary:")
    print(f"- Folders scanned: {total_folders}")
    print(f"- Total files found: {total_files}")
    print(f"- Loaded images: {loaded_files}")
    print(f"- Skipped files: {skipped_files}")
    print(f"- Unique labels: {set(labels) if labels else set()}")
    
    if loaded_files == 0:
        raise ValueError("No images loaded! Check formats, paths, or run diagnostic commands.")
    
    return np.array(images), np.array(labels)

def save_dataset(images, labels, output_dir='data/processed'):
    """
    Save preprocessed dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'images.npy'), images)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    print(f"Dataset saved to {output_dir}")