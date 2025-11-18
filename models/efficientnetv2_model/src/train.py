import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import yaml, json
import numpy as np
from .model import create_model, EMA

class FaceExpressionCropDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=320, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Support YOLO format: root_dir/images/split/*.jpg and root_dir/labels/split/*.txt
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
        from PIL import Image
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(split='train', img_size=320):
    if split=='train':
        return transforms.Compose([
            transforms.Resize((img_size+32, img_size+32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            transforms.RandomErasing(p=0.15)
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def train(dataset_root='dataset_cls', num_classes=7, epochs=100, batch_size=32, lr=1e-4, img_size=320, patience=30, device_type='mps'):
    if device_type=='cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_type=='mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    train_ds = FaceExpressionCropDataset(dataset_root, 'train', img_size, get_transforms('train', img_size))
    val_ds = FaceExpressionCropDataset(dataset_root, 'val', img_size, get_transforms('val', img_size))
    print(f"Train samples: {len(train_ds)} Val samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    model = create_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scaler = torch.cuda.amp.GradScaler() if device.type=='cuda' else None
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)*5
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1,warmup_steps)
        progress = (step - warmup_steps)/max(1,total_steps-warmup_steps)
        return max(0.05, 0.5*(1+np.cos(np.pi*progress)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, decay=0.999)
    best_acc = 0.0
    patience_counter = 0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss=0.0; correct=0; total=0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for i,(x,y) in enumerate(pbar):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss = criterion(out,y)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer); scaler.update()
            else:
                out = model(x)
                loss = criterion(out,y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
            scheduler.step()
            ema.update(model)
            running_loss += loss.item()
            pred = out.argmax(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
            pbar.set_postfix({'loss':f'{running_loss/(i+1):.3f}','acc':f'{100*correct/total:.2f}'})
        train_acc = 100*correct/total
        # Validation with EMA weights
        ema.apply_to(model)
        model.eval(); val_loss=0.0; vcorrect=0; vtotal=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out,y)
                val_loss += loss.item()
                vpred = out.argmax(1)
                vtotal += y.size(0)
                vcorrect += vpred.eq(y).sum().item()
        val_acc = 100*vcorrect/vtotal
        print(f'Epoch {epoch}: Train Acc {train_acc:.2f}% Val Acc {val_acc:.2f}% LR {optimizer.param_groups[0]["lr"]:.6f}')
        # Restore model weights (already EMA applied permanently for next epoch updates)
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            ckpt_dir = Path('models/efficientnetv2_model/checkpoints'); ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'val_acc':val_acc}, ckpt_dir/'best.pth')
            print(f'✅ Saved best model (Val {val_acc:.2f}%)')
        else:
            patience_counter +=1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}. Best Acc {best_acc:.2f}%')
            break
    print(f'Training complete. Best Val Acc {best_acc:.2f}%')

def main():
    # Calculate project root and dataset path
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATASET_PATH = PROJECT_ROOT / 'dataset'
    
    train(dataset_root=str(DATASET_PATH))

if __name__=='__main__':
    main()
