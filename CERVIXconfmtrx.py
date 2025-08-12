from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import argparse
import itertools
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from models.clip import clip
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models
from cervix_dataset import CervixDataset  # <-- Updated Dataset class

# -------------------
# Argument Parser
# -------------------
parser = argparse.ArgumentParser(description='PyTorch CLIP + CustomNet Testing on Cervix Dataset')
parser.add_argument('--model', type=str, default='Ourmodel', help='CNN architecture')
parser.add_argument('--mode', type=int, default=1, help='Feature mode (0=Image only, 1=Image+Text)')
parser.add_argument('--bs', default=64, type=int, help='Batch size')
parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--file_path', type=str, default='cervix_data_224.csv', help='CSV file path')
parser.add_argument('--checkpoint', type=str, default='best_cervix_model.pth', help='Path to checkpoint')
opt = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------
# Transforms
# -------------------
transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
])

# -------------------
# Class Names for Cervix Dataset
# -------------------
class_names = [
    "HSIL", "LSIL", "NIM", "SCC"
]
class_labels = list(range(len(class_names)))

# Full class descriptions for better understanding
class_descriptions = {
    "HSIL": "High-grade Squamous Intraepithelial Lesion",
    "LSIL": "Low-grade Squamous Intraepithelial Lesion", 
    "NIM": "Normal/Negative for Intraepithelial Malignancy",
    "SCC": "Squamous Cell Carcinoma"
}

print("Cervical Histopathology Classification:")
for i, class_name in enumerate(class_names):
    print(f"  {i}: {class_name} - {class_descriptions[class_name]}")

# -------------------
# Confusion Matrix Plot
# -------------------
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

# -------------------
# Custom Model
# -------------------
class CustomNet(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------
# Load CLIP
# -------------------
print("Loading CLIP model...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()
clip_model.eval()
print("CLIP model loaded successfully")

# -------------------
# Load CustomNet
# -------------------
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

if opt.mode == 0:
    print("Using Image-only features (512-dim)")
    net = CustomNet(num_classes=num_classes, feature_dim=512)
elif opt.mode == 1:
    print("Using Image+Text features (1024-dim -> 512-dim after pooling)")
    net = CustomNet(num_classes=num_classes, feature_dim=512)
else:
    net = nn.Sequential(nn.ReLU(), nn.Linear(512, num_classes))

# Load checkpoint
print(f"Loading checkpoint from: {opt.checkpoint}")
try:
    checkpoint = torch.load(opt.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['net'])
    print("Checkpoint loaded successfully")
except FileNotFoundError:
    print(f"Error: Checkpoint file '{opt.checkpoint}' not found!")
    print("Please ensure the checkpoint file exists or train a model first.")
    exit(1)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit(1)

net.to(device)
net.eval()

# -------------------
# Load Dataset
# -------------------
print(f"Loading test dataset from: {opt.file_path}")
try:
    testset = CervixDataset(split='Testing', transform=transforms_valid, file_path=opt.file_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)
    print(f"Test dataset loaded: {len(testset)} samples, {len(testloader)} batches")
except FileNotFoundError:
    print(f"Error: CSV file '{opt.file_path}' not found!")
    print("Please ensure the CSV file exists or run the dataset preprocessing script first.")
    exit(1)

# -------------------
# Evaluation
# -------------------
print("Starting evaluation...")
correct = 0
total = 0
overall_conf_matrix = np.zeros((num_classes, num_classes))

all_predicted = []
all_targets = []

with torch.no_grad():
    for batch_idx, (images, captions, labels) in enumerate(testloader):
        images, captions, labels = images.to(device), captions.to(device), labels.to(device)

        # Extract image features
        image_features = clip_model.encode_image(images)
        
        if opt.mode == 1:
            # Extract text features and combine with image features
            text_features = clip_model.encode_text(captions)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            features = torch.cat((image_features, text_features), dim=1)
        else:
            # Use only image features
            features = image_features

        # Apply pooling to reduce dimensionality
        pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2)
        features = features.unsqueeze(1)  # Add channel dimension for pooling
        features = pooling_layer(features).squeeze(1)
        features = features.view(features.size(0), -1).float()

        # Get predictions
        outputs = net(features)
        _, predicted = torch.max(outputs.data, 1)
        
        # Update metrics
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()

        # Update confusion matrix
        conf_matrix = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), labels=class_labels)
        overall_conf_matrix += conf_matrix

        # Store predictions and targets
        all_predicted.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(testloader)} processed")

# -------------------
# Results and Metrics
# -------------------
acc = 100. * correct / total
print(f"\n{'='*50}")
print(f"FINAL RESULTS")
print(f"{'='*50}")
print(f"Test Accuracy: {acc:.3f}% ({correct}/{total})")
print(f"\nPer-class Results:")

# Detailed classification report
print('\nClassification Report:')
report = classification_report(all_targets, all_predicted, target_names=class_names, digits=4)
print(report)

# Save detailed results
with open('cervix_test_results.txt', 'w') as f:
    f.write(f"Cervix Dataset Test Results\n")
    f.write(f"{'='*30}\n")
    f.write(f"Test Accuracy: {acc:.3f}% ({correct}/{total})\n")
    f.write(f"Mode: {'Image+Text' if opt.mode == 1 else 'Image-only'}\n")
    f.write(f"Checkpoint: {opt.checkpoint}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# -------------------
# Plot and Save Confusion Matrix
# -------------------
plt.figure(figsize=(10, 8))
plot_confusion_matrix(overall_conf_matrix, classes=class_names, normalize=False,
                      title=f'Cervix Dataset Confusion Matrix (Accuracy: {acc:.3f}%)')
plt.savefig('confusion_matrix_cervix.png', dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved as 'confusion_matrix_cervix.png'")

# Also plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(overall_conf_matrix, classes=class_names, normalize=True,
                      title=f'Cervix Dataset Normalized Confusion Matrix (Accuracy: {acc:.3f}%)')
plt.savefig('confusion_matrix_cervix_normalized.png', dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix saved as 'confusion_matrix_cervix_normalized.png'")

plt.show()

print(f"\nTesting completed successfully!")
print(f"Results saved to 'cervix_test_results.txt'")
