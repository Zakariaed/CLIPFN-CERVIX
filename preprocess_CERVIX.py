from __future__ import print_function
from PIL import Image
import torch
import clip
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split

# Load CLIP model and preprocessing
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

class CervixDataset(data.Dataset):
    def __init__(self, split='Training', transform=None, file_path=None, test_size=0.2, random_state=42):
        self.transform = transform
        self.split = split
        self.file_path = file_path
        
        # Read CSV (images, text, labels)
        self.data = pd.read_csv(self.file_path)
        
        # Print dataset info
        print(f"Total dataset size: {len(self.data)}")
        print(f"Classes: {sorted(self.data['labels'].unique())}")
        
        # Class distribution
        class_counts = self.data['labels'].value_counts().sort_index()
        print("Class distribution:")
        class_names = ['HSIL', 'LSIL', 'NIM', 'SCC']  # Based on label indices 0,1,2,3
        for idx, count in class_counts.items():
            print(f"  {class_names[idx]} (label {idx}): {count} images")
        
        # Stratified train/test split
        train_df, test_df = train_test_split(
            self.data,
            test_size=test_size,
            stratify=self.data['labels'],
            random_state=random_state
        )
        
        if self.split == 'Training':
            self.dataset = train_df.reset_index(drop=True)
        else:
            self.dataset = test_df.reset_index(drop=True)
            
        print(f"{self.split} set size: {len(self.dataset)}")
        
        # Print split class distribution
        split_class_counts = self.dataset['labels'].value_counts().sort_index()
        print(f"{self.split} set class distribution:")
        for idx, count in split_class_counts.items():
            print(f"  {class_names[idx]} (label {idx}): {count} images")
    
    def __getitem__(self, index):
        row = self.dataset.iloc[index]
        image_path = row['images']
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        text = row['text']
        caption = clip.tokenize([text])  # Tokenized text for CLIP
        label = torch.tensor(row['labels'], dtype=torch.long)
        
        return image, caption.squeeze(0), label
    
    def __len__(self):
        return len(self.dataset)

# Example usage:
if __name__ == "__main__":
    # Create training dataset
    train_dataset = CervixDataset(
        split='Training',
        transform=preprocess,
        file_path='./cervix_data_224.csv',
        test_size=0.2,
        random_state=42
    )
    
    # Create test dataset
    test_dataset = CervixDataset(
        split='Testing',
        transform=preprocess,
        file_path='./cervix_data_224.csv',
        test_size=0.2,
        random_state=42
    )
    
    # Example: Get a sample
    image, caption, label = train_dataset[0]
    print(f"\nSample data shapes:")
    print(f"Image shape: {image.shape}")
    print(f"Caption shape: {caption.shape}")
    print(f"Label: {label}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"\nDataLoaders created:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
