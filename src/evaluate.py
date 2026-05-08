import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, f1_score
from src.fusion_model import MultimodalFusion
from src.data_loader import get_data_loaders

def evaluate_model(model, val_loader, device, num_classes):
    """
    Runs evaluation on the validation set to get true labels and predictions.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Running final evaluation on Validation Set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(images, input_ids, attention_mask)
            
            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    return all_preds, all_targets

def plot_roc_curves(y_true, y_pred, classes, top_k=10, save_path="outputs/roc_curves.png"):
    """
    Plots ROC curves for the top_k most frequent diseases.
    """
    # Find the top K most frequent classes based on ground truth
    class_frequencies = y_true.sum(axis=0)
    top_indices = class_frequencies.argsort()[-top_k:][::-1]
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', top_k)
    
    for i, idx in enumerate(top_indices):
        class_name = classes[idx]
        
        # Calculate ROC for this specific class
        # Skip if a class has no positive instances in the validation set
        if np.sum(y_true[:, idx]) == 0:
            continue
            
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred[:, idx])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors(i), lw=2, 
                 label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for Top {top_k} Most Frequent Classes', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\nSaved ROC curves to {save_path}")
        
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load labels
    labels_csv_path = 'data/processed/labels.csv'
    dataset_csv_path = 'data/processed/dataset.csv'
    model_path = 'outputs/models/best_model_V2.pth'
    
    df_labels = pd.read_csv(labels_csv_path)
    classes = df_labels['class_name'].tolist()
    num_classes = len(classes)
    
    # 2. Load the best model
    model = MultimodalFusion(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 3. Get DataLoaders
    # Use larger batch size for evaluation
    _, val_loader, _ = get_data_loaders(dataset_csv_path, batch_size=16)
    
    # 4. Evaluate
    y_pred, y_true = evaluate_model(model, val_loader, device, num_classes)
    
    # Calculate metrics (Threshold = 0.5)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision_micro = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    precision_macro = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS (Validation Set)")
    print("="*40)
    print(f"Exact Match Accuracy: {accuracy:.4f}")
    print("\n--- Micro-Averaged (Overall Performance) ---")
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall:    {recall_micro:.4f}")
    print(f"Micro F1-Score:  {f1_micro:.4f}")
    print("\n--- Macro-Averaged (Class-Level Performance) ---")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1-Score:  {f1_macro:.4f}")
    print("="*40 + "\n")
    
    # 5. Plot Results
    plot_roc_curves(y_true, y_pred, classes, top_k=10)
