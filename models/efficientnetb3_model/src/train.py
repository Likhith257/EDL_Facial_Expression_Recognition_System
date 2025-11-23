
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from model import create_model


class FacialExpressionDataset(Dataset):
    """Dataset for facial expression recognition"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load images and labels
        self.images = []
        self.labels = []
        
        img_dir = self.root_dir / 'images' / split
        label_dir = self.root_dir / 'labels' / split
        
        if img_dir.exists():
            for img_file in sorted(img_dir.glob('*.jpg')):
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    # Read YOLO format label (class is first value)
                    with open(label_file, 'r') as f:
                        label = int(f.readline().split()[0])
                    self.images.append(img_file)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train', img_size=224):
    """Get data transforms"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),  # Increased from 15
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10),  # Added shear
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),  # Increased
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # NEW: perspective transform
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))  # Increased probability
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def train(config_path=None, **kwargs):
    """Main training function"""
    
    # Load config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with kwargs
    config.update(kwargs)
    
    # Set defaults
    dataset_path = config.get('dataset_path', '../../dataset')
    num_classes = config.get('num_classes', 7)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('epochs', 100)
    lr = config.get('lr', 0.001)
    img_size = config.get('img_size', 224)
    patience = config.get('patience', 15)
    device_type = config.get('device', 'mps')
    
    # Setup device
    if device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using Apple M4 Pro GPU (MPS)")
    elif device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (training will be slower)")
    
    # Create datasets
    print(f"\n📁 Loading dataset from: {dataset_path}")
    train_dataset = FacialExpressionDataset(
        dataset_path, split='train', 
        transform=get_transforms('train', img_size)
    )
    val_dataset = FacialExpressionDataset(
        dataset_path, split='val',
        transform=get_transforms('val', img_size)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    print(f"\n🏗️  Building EfficientNet-B3 model...")
    model = create_model(num_classes=num_classes, pretrained=True, device=device_type)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # Increased from 0.1
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))  # Reduced weight decay
    
    # Warmup + Cosine scheduler
    warmup_epochs = 5
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'checkpoints'
    output_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler per batch (for warmup)
        if epoch <= warmup_epochs:
            for _ in range(len(train_loader)):
                scheduler.step()
        else:
            scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
            break
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {output_dir}")
    
    return model, history


def main():
    """Main execution"""
    # Project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATASET_PATH = PROJECT_ROOT / 'dataset'
    CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
    
    train(
        config_path=CONFIG_PATH if CONFIG_PATH.exists() else None,
        dataset_path=str(DATASET_PATH),
        num_classes=7,
        batch_size=48,  # Increased from 32 for better gradient estimates
        epochs=120,  # Increased from 100
        lr=0.00015,  # Increased from 0.0001
        img_size=224,
        patience=25,  # Reduced from 30 for faster convergence
        device='mps'
    )


if __name__ == "__main__":
    main()
