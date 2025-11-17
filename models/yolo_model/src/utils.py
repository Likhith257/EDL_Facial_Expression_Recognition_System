import cv2
import numpy as np
from pathlib import Path
import json
import yaml


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path='config.yaml'):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def visualize_bbox(image, bbox, label, confidence, color=(0, 255, 0)):
    """
    Visualize bounding box on image.
    
    Args:
        image: Input image (BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        label: Class label
        confidence: Confidence score
        color: Box color (BGR)
    
    Returns:
        Annotated image
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Create label text
    text = f"{label}: {confidence:.2f}"
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    
    # Draw background for text
    cv2.rectangle(
        image,
        (x1, y1 - text_height - 10),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )
    
    return image


def resize_image(image, target_size=640):
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size (longest side)
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= target_size:
        return image
    
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def create_emotion_legend():
    """
    Create emotion color legend.
    
    Returns:
        Legend image
    """
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
    colors = [
        (0, 0, 255),      # Angry - Red
        (0, 165, 255),    # Disgust - Orange
        (128, 0, 128),    # Fear - Purple
        (0, 255, 0),      # Happy - Green
        (255, 255, 255),  # Neutral - White
        (255, 0, 0),      # Sad - Blue
        (0, 255, 255),    # Surprised - Yellow
    ]
    
    # Create legend image
    legend_height = 30 * len(emotions)
    legend_width = 200
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 240
    
    for i, (emotion, color) in enumerate(zip(emotions, colors)):
        y = 30 * i + 20
        
        # Draw color box
        cv2.rectangle(legend, (10, y - 15), (40, y + 5), color, -1)
        cv2.rectangle(legend, (10, y - 15), (40, y + 5), (0, 0, 0), 1)
        
        # Draw text
        cv2.putText(
            legend,
            emotion,
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    return legend


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: First box (x1, y1, x2, y2)
        box2: Second box (x1, y1, x2, y2)
    
    Returns:
        IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def save_results(results, output_path):
    """
    Save detection results to JSON file.
    
    Args:
        results: Detection results
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)


def load_results(input_path):
    """
    Load detection results from JSON file.
    
    Args:
        input_path: Input file path
    
    Returns:
        Detection results
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    return results


def check_gpu():
    """
    Check if GPU is available.
    
    Returns:
        GPU information
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            return {
                'available': True,
                'type': 'CUDA',
                'count': gpu_count,
                'name': gpu_name,
                'memory_gb': gpu_memory
            }
        elif torch.backends.mps.is_available():
            return {
                'available': True,
                'type': 'MPS',
                'name': 'Apple Silicon GPU',
                'count': 1
            }
        else:
            return {'available': False}
    except ImportError:
        return {'available': False, 'error': 'PyTorch not installed'}


def print_system_info():
    """Print system information."""
    import platform
    import sys
    
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # GPU info
    gpu_info = check_gpu()
    if gpu_info['available']:
        print(f"GPU: {gpu_info['name']} ({gpu_info['type']})")
        if 'memory_gb' in gpu_info:
            print(f"GPU Memory: {gpu_info['memory_gb']:.2f} GB")
    else:
        print("GPU: Not available (using CPU)")
    
    print("=" * 60)


if __name__ == "__main__":
    print_system_info()
