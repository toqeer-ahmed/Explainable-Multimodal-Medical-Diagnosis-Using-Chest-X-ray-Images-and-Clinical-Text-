import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

from src.data_loader import get_data_loaders
from src.fusion_model import MultimodalFusion
import torch.nn.functional as F

# --- UPGRADE: Focal Loss to handle Extreme Imbalance ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # Progress bar for training
    loop = tqdm(loader, total=len(loader), desc="Training", leave=False)
    
    for batch in loop:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, input_ids, attention_mask)
        
        # Loss computation (BCEWithLogitsLoss applies sigmoid internally)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    loop = tqdm(loader, total=len(loader), desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in loop:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())
            
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Threshold predictions at 0.5 for binary classification metrics
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate Metrics (using macro average for multi-label)
    # Wrap in try-except in case some classes have no positive samples in validation split
    try:
        f1 = f1_score(all_targets, binary_preds, average='macro', zero_division=0)
        # Note: ROC AUC requires both classes present. For a quick script, we skip it if it fails.
        roc_auc = roc_auc_score(all_targets, all_preds, average='macro', multi_class='ovr')
    except ValueError:
        roc_auc = 0.0
        
    return running_loss / len(loader), f1, roc_auc

def main(csv_path, epochs=10, batch_size=32, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Get DataLoaders
    print("Loading data...")
    train_loader, val_loader, num_classes = get_data_loaders(csv_path, batch_size=batch_size, num_workers=2)
    print(f"Number of classes: {num_classes}")
    
    # 2. Initialize Model
    print("Initializing Multimodal Fusion Model...")
    model = MultimodalFusion(num_classes=num_classes)
    model = model.to(device)
    
    # 3. Setup Loss and Optimizer
    # --- UPGRADE: Using Focal Loss instead of standard BCE ---
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # --- UPGRADE: Learning Rate Scheduler ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 4. Training Loop
    best_f1 = 0.0
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    
    # List to store history for plotting
    history = []
    
    print("Starting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_auc = eval_epoch(model, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val F1-Score: {val_f1:.4f} | Val ROC-AUC: {val_auc:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'val_auc': val_auc
        })
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join("outputs", "models", "best_model.pth"))
            print("--> Saved new best model!")
            
    # Save training history to CSV
    import pandas as pd
    pd.DataFrame(history).to_csv(os.path.join("outputs", "training_history.csv"), index=False)
    print("--> Training history saved to outputs/training_history.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Multimodal Model")
    parser.add_argument("--csv_path", type=str, default="data/processed/dataset.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    args = parser.parse_args()
    main(args.csv_path, args.epochs, args.batch_size, args.lr)
