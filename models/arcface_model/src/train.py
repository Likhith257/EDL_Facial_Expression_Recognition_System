import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import argparse

class FaceDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        # Filter for jpg images
        self.img_names = sorted([f.name for f in self.img_dir.glob('*.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.img_dir / img_name
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        # Read label
        label_path = self.label_dir / f"{Path(img_name).stem}.txt"
        label = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                # Assuming YOLO format: class x y w h
                content = f.read().strip().split()
                if content:
                    label = int(content[0])
        
        return image, label

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.s = s
        self.m = m

    def forward(self, x, labels=None):
        x_norm = nn.functional.normalize(x, p=2, dim=1)
        w_norm = nn.functional.normalize(self.fc.weight, p=2, dim=1)
        logits = torch.matmul(x_norm, w_norm.t())
        
        if labels is not None:
            theta = torch.acos(torch.clamp(logits, -1.0, 1.0))
            target_logit = torch.cos(theta + self.m)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.view(-1,1), 1.0)
            logits = logits * (1 - one_hot) + target_logit * one_hot
            
        logits *= self.s
        return logits

def train(epochs=10, batch_size=32, lr=1e-4, device_type='mps'):
    # Setup device
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using CUDA GPU")
    elif device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using Apple M4 Pro GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU")

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_ROOT = PROJECT_ROOT / 'dataset' / 'images' / 'train'
    LABEL_ROOT = PROJECT_ROOT / 'dataset' / 'labels' / 'train'
    MODEL_SAVE_PATH = PROJECT_ROOT / 'models' / 'arcface_model' / 'arcface_model.pt'
    
    if not DATA_ROOT.exists():
        print(f"❌ Dataset not found at {DATA_ROOT}")
        return

    # Transforms with augmentation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),  # Increased from 112
        transforms.RandomCrop((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset & Loader
    print("📁 Loading dataset...")
    train_set = FaceDataset(DATA_ROOT, LABEL_ROOT, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"   Found {len(train_set)} images")

    # Model
    print("🏗️  Building ArcFace model...")
    backbone = models.resnet18(pretrained=True)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    
    num_classes = 7 
    arcface_head = ArcFaceHead(num_features, num_classes)
    
    # Move to device
    backbone = backbone.to(device)
    arcface_head = arcface_head.to(device)
    
    # Optimizer with AdamW and weight decay
    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(arcface_head.parameters()), 
        lr=lr, 
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    warmup_epochs = 3
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Loop
    print(f"🎯 Starting training for {epochs} epochs...")
    best_acc = 0.0
    step = 0
    
    for epoch in range(epochs):
        backbone.train()
        arcface_head.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training for CUDA
            if scaler:
                with torch.cuda.amp.autocast():
                    features = backbone(images)
                    logits = arcface_head(features, labels)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(arcface_head.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                features = backbone(images)
                logits = arcface_head(features, labels)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(arcface_head.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            step += 1
            running_loss += loss.item()
            
            # Accuracy
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({
                'backbone': backbone.state_dict(),
                'head': arcface_head.state_dict(),
                'epoch': epoch,
                'accuracy': epoch_acc
            }, MODEL_SAVE_PATH)
            print(f'  ✅ Saved best model (Acc: {epoch_acc:.2f}%)')

    # Save final model
    torch.save({
        'backbone': backbone.state_dict(),
        'head': arcface_head.state_dict()
    }, MODEL_SAVE_PATH.parent / 'arcface_model_final.pt')
    print(f'✅ Final model saved to {MODEL_SAVE_PATH.parent / "arcface_model_final.pt"}')
    print(f'🏆 Best accuracy: {best_acc:.2f}%')

def main():
    train(epochs=50, batch_size=64, lr=0.0005, device_type='mps')  # Optimized hyperparameters

if __name__ == "__main__":
    main()
