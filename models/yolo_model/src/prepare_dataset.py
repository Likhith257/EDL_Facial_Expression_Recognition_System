import os
import re
import cv2
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

_THIS_FILE = Path(__file__).resolve()
YOLO_ROOT = _THIS_FILE.parents[1]
PROJECT_ROOT = _THIS_FILE.parents[3]

# Emotion mapping
EMOTION_MAPPING = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6,
    'AN': 0,  # Additional mappings
    'DI': 1,
    'FE': 2,
    'HA': 3,
    'NE': 4,
    'SA': 5,
    'SU': 6
}


class DatasetPreparer:
    def __init__(self, source_dir, output_dir):
        """
        Initialize the dataset preparer.
        
        Args:
            source_dir: Source directory containing student datasets
            output_dir: Output directory for organized YOLO dataset
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'faces_detected': 0,
            'skipped_images': 0,
            'emotions': {k: 0 for k in EMOTION_MAPPING.keys() if len(k) > 2}
        }
    
    def normalize_emotion_name(self, name):
        """Normalize various emotion naming conventions."""
        name_lower = name.lower()
        
        # Direct mapping
        if name_lower in EMOTION_MAPPING:
            return name_lower
        
        # Check for partial matches
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']:
            if emotion in name_lower or name_lower in emotion:
                return emotion
        
        # Check for abbreviated forms
        name_upper = name.upper()
        if name_upper in EMOTION_MAPPING:
            emotion_id = EMOTION_MAPPING[name_upper]
            for k, v in EMOTION_MAPPING.items():
                if v == emotion_id and len(k) > 2:
                    return k
        
        return None

    def infer_emotion_from_path(self, img_path: Path):
        """Infer emotion label from parent folders or filename tokens.

        Priority:
        1) Any parent folder that maps to an emotion
        2) Abbreviation tokens in filename like -FE-, (HA), _NE_, etc.
        3) Full emotion words in filename
        Returns normalized full-name emotion or None.
        """
        # 1) From parent folders
        for parent in img_path.parents:
            # stop at project root
            if parent == PROJECT_ROOT:
                break
            em = self.normalize_emotion_name(parent.name)
            if em:
                return em

        # 2) From filename tokens (abbreviations)
        stem = img_path.stem
        name_upper = stem.upper()
        # Look for standalone tokens between separators or in parentheses
        # e.g., -FE-, _HA_, (NE), 01-SA-03
        token_match = re.search(r"(?<![A-Z0-9])(AN|DI|FE|HA|NE|SA|SU)(?![A-Z0-9])", name_upper)
        if token_match:
            abbr = token_match.group(1)
            # Map to full name using EMOTION_MAPPING
            emotion_id = EMOTION_MAPPING.get(abbr)
            if emotion_id is not None:
                for k, v in EMOTION_MAPPING.items():
                    if v == emotion_id and len(k) > 2:
                        return k

        # 3) Full emotion words in filename
        name_lower = stem.lower()
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']:
            if emotion in name_lower:
                return emotion

        return None
    
    def detect_face(self, image):
        """
        Detect face in image and return bounding box in YOLO format.
        
        Returns:
            tuple: (x_center, y_center, width, height) normalized to [0, 1]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Take the largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        x, y, w, h = faces[0]
        img_h, img_w = image.shape[:2]
        
        # Convert to YOLO format (normalized center coordinates + width/height)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h
        
        return x_center, y_center, width, height
    
    def process_image(self, img_path, emotion_label, output_img_dir, output_label_dir):
        """Process a single image and create YOLO annotation."""
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                self.stats['skipped_images'] += 1
                return False
            
            # Detect face
            bbox = self.detect_face(image)
            if bbox is None:
                self.stats['skipped_images'] += 1
                return False
            
            # Get emotion class
            emotion_class = EMOTION_MAPPING.get(emotion_label)
            if emotion_class is None:
                self.stats['skipped_images'] += 1
                return False
            
            # Create unique filename
            img_name = f"{emotion_label}_{self.stats['processed_images']}.jpg"
            
            # Save image
            img_output_path = output_img_dir / img_name
            cv2.imwrite(str(img_output_path), image)
            
            # Create YOLO annotation
            label_name = img_name.replace('.jpg', '.txt')
            label_path = output_label_dir / label_name
            
            with open(label_path, 'w') as f:
                # YOLO format: class x_center y_center width height
                f.write(f"{emotion_class} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            self.stats['processed_images'] += 1
            self.stats['faces_detected'] += 1
            self.stats['emotions'][emotion_label] += 1
            
            return True
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            self.stats['skipped_images'] += 1
            return False
    
    def scan_directory(self, directory):
        """Scan directory for images and infer emotion labels.

        Includes images either placed under emotion-named folders or whose
        filenames contain recognizable emotion abbreviations/words.
        """
        image_files = []
        VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.tiff'}

        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in VALID_EXT:
                    img_path = Path(root) / file
                    emotion = self.infer_emotion_from_path(img_path)
                    if emotion:
                        image_files.append((img_path, emotion))
                    self.stats['total_images'] += 1

        return image_files
    
    def prepare_dataset(self, test_size=0.2, val_size=0.1):
        """
        Main method to prepare the complete dataset.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
        """
        print("🔍 Scanning source directory...")
        image_files = self.scan_directory(self.source_dir)
        
        if not image_files:
            print("❌ No images found! Check your source directory.")
            return
        
        print(f"📊 Found {len(image_files)} total images")
        
        # Split dataset
        train_val_files, test_files = train_test_split(
            image_files, test_size=test_size, random_state=42, 
            stratify=[x[1] for x in image_files]
        )
        
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_size, random_state=42,
            stratify=[x[1] for x in train_val_files]
        )
        
        print(f"📚 Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Process each split
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            print(f"\n⚙️  Processing {split_name} set...")
            img_dir = self.output_dir / 'images' / split_name
            label_dir = self.output_dir / 'labels' / split_name
            
            for img_path, emotion in tqdm(files, desc=split_name):
                self.process_image(img_path, emotion, img_dir, label_dir)
        
        # Save dataset statistics
        self.save_statistics()
        
        # Create data.yaml for YOLO
        self.create_data_yaml()
        
        print("\n✅ Dataset preparation complete!")
        self.print_statistics()
    
    def save_statistics(self):
        """Save dataset statistics to file."""
        stats_path = self.output_dir / 'dataset_stats.txt'
        with open(stats_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("DATASET PREPARATION STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total images scanned: {self.stats['total_images']}\n")
            f.write(f"Successfully processed: {self.stats['processed_images']}\n")
            f.write(f"Faces detected: {self.stats['faces_detected']}\n")
            f.write(f"Skipped images: {self.stats['skipped_images']}\n\n")
            f.write("Emotion Distribution:\n")
            f.write("-" * 50 + "\n")
            for emotion, count in sorted(self.stats['emotions'].items()):
                f.write(f"{emotion.capitalize():15s}: {count:5d}\n")
    
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total images scanned: {self.stats['total_images']}")
        print(f"Successfully processed: {self.stats['processed_images']}")
        print(f"Faces detected: {self.stats['faces_detected']}")
        print(f"Skipped images: {self.stats['skipped_images']}")
        print(f"\nEmotion Distribution:")
        print("-" * 50)
        for emotion, count in sorted(self.stats['emotions'].items()):
            print(f"{emotion.capitalize():15s}: {count:5d}")
    
    def create_data_yaml(self):
        """Create data.yaml file for YOLO training."""
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 7,
            'names': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"\n📄 Created data.yaml at: {yaml_path}")


def main():
    """Main execution function."""
    # Configuration: always reference project-level raw and output dataset
    SOURCE_DIR = str((PROJECT_ROOT / "DATASET - To Students").resolve())
    OUTPUT_DIR = str((PROJECT_ROOT / "dataset").resolve())
    
    print("🚀 Starting Dataset Preparation for YOLO")
    print("=" * 50)
    
    # Initialize preparer
    preparer = DatasetPreparer(SOURCE_DIR, OUTPUT_DIR)
    
    # Prepare dataset
    preparer.prepare_dataset(test_size=0.15, val_size=0.15)
    
    print(f"\n💾 Dataset saved to: {Path(OUTPUT_DIR).absolute()}")
    print("🎯 Ready for YOLO training!")


if __name__ == "__main__":
    main()
