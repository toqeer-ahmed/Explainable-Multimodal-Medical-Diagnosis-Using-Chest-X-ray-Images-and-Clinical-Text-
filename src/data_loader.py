import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='bert-base-uncased', max_len=128, transform=None):
        """
        Args:
            csv_file (string): Path to the processed dataset.csv file.
            tokenizer_name (string): Name of the HuggingFace tokenizer.
            max_len (int): Maximum sequence length for the text.
            transform (callable, optional): Optional transform to be applied on a sample image.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        
        # Get label columns (all columns after image_path and report_text)
        self.label_cols = self.data_frame.columns[2:].tolist()
        
        # Default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # ImageNet normalization
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Load Image
        img_path = self.data_frame.iloc[idx, 0]
        # Fix backslashes from Windows for Linux (Colab) compatibility
        img_path = img_path.replace('\\', '/')
        # Open image and convert to RGB (some X-rays are grayscale)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 2. Load Text
        text = str(self.data_frame.iloc[idx, 1])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # 3. Load Labels
        labels = self.data_frame.iloc[idx, 2:].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

def get_data_loaders(csv_file, batch_size=32, num_workers=4, train_split=0.8):
    """
    Utility function to get train and validation dataloaders.
    """
    dataset = MultimodalDataset(csv_file=csv_file)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, len(dataset.label_cols)

if __name__ == "__main__":
    # Quick test
    csv_path = os.path.join("data", "processed", "dataset.csv")
    if os.path.exists(csv_path):
        print("Testing DataLoader...")
        train_loader, val_loader, num_classes = get_data_loaders(csv_path, batch_size=4, num_workers=0)
        
        batch = next(iter(train_loader))
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Total number of classes: {num_classes}")
    else:
        print(f"Dataset not found at {csv_path}")
