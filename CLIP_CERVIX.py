from __future__ import print_function
import os
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import clip
from preprocess_CERVIX import CervixDataset      # <-- your CSV-based dataset
import utils

# -------------------
# Kaggle Environment Setup
# -------------------
def setup_kaggle_environment():
    """Setup environment variables and paths for Kaggle"""
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print("Running on Kaggle environment detected")
        # Set CUDA visible devices to use GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # Increase shared memory for DataLoader
        torch.multiprocessing.set_sharing_strategy('file_system')
    return True

setup_kaggle_environment()

# -------------------
# Argparse with Kaggle-friendly defaults
# -------------------
parser = argparse.ArgumentParser(description='PyTorch CLIP Training on Cervix Dataset')
parser.add_argument('--model', type=str, default='Ourmodel', help='CNN architecture')
parser.add_argument('--mode', type=int, default=1, help='Feature mode (0=image, 1=image+text)')
parser.add_argument('--dataset', type=str, default='Cervix_Ourmodel', help='dataset folder prefix')
parser.add_argument('--fold', default=1, type=int, help='(ignored for this dataset)')
# Reduced batch size for P100 memory constraints
parser.add_argument('--bs', default=4, type=int, help='batch_size (reduced for P100)')
parser.add_argument('--lr', default=0.003, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# Reduced workers for Kaggle
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--epochs', type=int, default=30)  # Reduced from 40 to 30
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--file_path', type=str, default='cervix_data_224.csv', help='CSV describing dataset')
parser.add_argument('--checkpoint', type=str, default='', help='optional checkpoint path to load')

# Parse args with Kaggle compatibility
try:
    opt = parser.parse_args()
except:
    # If running in Kaggle notebook, set default args
    opt = argparse.Namespace()
    opt.model = 'Ourmodel'
    opt.mode = 1
    opt.dataset = 'Cervix_Ourmodel'
    opt.fold = 1
    opt.bs = 4  # Reduced for P100
    opt.lr = 0.003
    opt.resume = False
    opt.workers = 2  # Reduced for Kaggle
    opt.epochs = 30  # Reduced from 40 to 30
    opt.weight_decay = 1e-4
    opt.file_path = 'cervix_data_224.csv'
    opt.checkpoint = ''

# GPU setup with error handling
use_cuda = torch.cuda.is_available()
if use_cuda:
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
device = "cuda:0" if use_cuda else "cpu"

# Clear GPU cache
if use_cuda:
    torch.cuda.empty_cache()

# -------------------
# Classes for Cervix Dataset
# -------------------
class_names = ["HSIL", "LSIL", "NIM", "SCC"]  # Updated for cervical histopathology
num_classes = len(class_names)
path = os.path.join(opt.dataset, str(opt.fold))

print("Cervical Histopathology Classification:")
print("Classes:", class_names)
print(f"Number of classes: {num_classes}")

# -------------------
# Memory-efficient transforms
# -------------------
transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(30),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Added for medical images
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
])

transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
])

# -------------------
# Memory-efficient Custom network
# -------------------
class CustomNet(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CustomNet, self).__init__()
        # Reduced hidden layer size for P100
        self.fc1 = nn.Linear(feature_dim, 256)  # Reduced from 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)  # Reduced from 512

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, accuracy):
        score = accuracy
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# -------------------
# Load CLIP with memory management
# -------------------
print("Loading CLIP model...")
try:
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    clip_model.eval()
    print("CLIP model loaded successfully")
except Exception as e:
    print(f"Error loading CLIP: {e}")
    raise

# Clear cache after loading CLIP
if use_cuda:
    torch.cuda.empty_cache()

def infer_feature_dim(clip_model, device, mode):
    """
    Run a tiny forward pass with dummy inputs to infer flattened feature size
    after the avgpool operations used in train/test.
    """
    clip_model.eval()
    with torch.no_grad():
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        dummy_text = clip.tokenize(["dummy"]).to(device)
        
        try:
            image_features = clip_model.encode_image(dummy_image)           # (1, D)
            image_features = image_features.unsqueeze(1)                    # (1,1,D)
            image_features = nn.AvgPool1d(kernel_size=2)(image_features).squeeze(1)  # (1, D')
            
            if mode == 1:
                text_features = clip_model.encode_text(dummy_text)         # (1, D_text)
                text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
                features = torch.cat((image_features, text_features), dim=1)  # (1, D_sum)
            else:
                features = image_features
                
            features = nn.AvgPool1d(kernel_size=2, stride=2)(features)     # 1D pooling
            flat = features.view(features.size(0), -1)
            return flat.size(1)
        except Exception as e:
            print(f"Error in feature dimension inference: {e}")
            return 512  # fallback dimension

feature_dim = infer_feature_dim(clip_model, device, opt.mode)
print(f"Inferred flattened feature dim: {feature_dim}")

# -------------------
# Instantiate network
# -------------------
if opt.mode == 0:
    net = CustomNet(num_classes=num_classes, feature_dim=feature_dim)
    print("Using Image-only features")
elif opt.mode == 1:
    net = CustomNet(num_classes=num_classes, feature_dim=feature_dim)
    print("Using Image+Text features")
else:
    net = nn.Sequential(nn.ReLU(), nn.Linear(feature_dim, num_classes))

net = net.to(device)
print(f"Network moved to {device}")

# -------------------
# DataLoaders with memory optimization
# -------------------
print("Preparing data loaders...")
try:
    trainset = CervixDataset(split='Training', transform=transforms_train, file_path=opt.file_path)
    testset  = CervixDataset(split='Testing',  transform=transforms_valid, file_path=opt.file_path)
    
    # Memory-efficient data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=opt.bs, 
        shuffle=True, 
        num_workers=opt.workers,
        pin_memory=True if use_cuda else False,
        persistent_workers=True if opt.workers > 0 else False
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=opt.bs, 
        shuffle=False, 
        num_workers=opt.workers,
        pin_memory=True if use_cuda else False,
        persistent_workers=True if opt.workers > 0 else False
    )
    
    print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}")
    print(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}")
except Exception as e:
    print(f"Error creating data loaders: {e}")
    raise

# -------------------
# Criterion & Optimizer with gradient clipping
# -------------------
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

# Optionally resume
best_Test_acc = 0.0
start_epoch = 0
if opt.resume and opt.checkpoint and os.path.exists(opt.checkpoint):
    print("Loading checkpoint:", opt.checkpoint)
    try:
        ck = torch.load(opt.checkpoint, map_location=device)
        net.load_state_dict(ck['net'])
        best_Test_acc = ck.get('best_Test_acc', 0.0)
        start_epoch = ck.get('best_Test_acc_epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch} with best acc: {best_Test_acc:.3f}%")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

# -------------------
# Memory monitoring function
# -------------------
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# -------------------
# Training & Testing functions with memory management
# -------------------
early_stopping = EarlyStopping(patience=8, delta=0.001)  # Slightly reduced patience for 30 epochs
total_processing_time_train = 0.0
total_processing_time_test = 0.0

def train(epoch):
    global total_processing_time_train
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    avg_pool = nn.AvgPool1d(kernel_size=2)

    all_labels = []
    all_predictions = []

    start_time = time.monotonic()
    
    for idx, (images, captions, labels) in enumerate(trainloader):
        try:
            batch_start = time.time()
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
            image_features = image_features.unsqueeze(1)
            image_features = avg_pool(image_features).squeeze(1)

            if opt.mode == 1:
                with torch.no_grad():
                    text_features = clip_model.encode_text(captions)
                    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
                features = torch.cat((image_features, text_features), dim=1)
            else:
                features = image_features

            features = nn.AvgPool1d(kernel_size=2, stride=2)(features)
            features = features.view(features.size(0), -1).float()

            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            batch_end = time.time()
            total_processing_time_train += (batch_end - batch_start)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if idx % 100 == 0:  # Periodic memory cleanup
                torch.cuda.empty_cache()

            utils.progress_bar(idx, len(trainloader),
                               'TrainLoss: %.4f | TrainAcc: %.3f%% (%d/%d)' %
                               (running_loss / (idx + 1), 100. * correct / total, correct, total))
                               
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM at batch {idx}, clearing cache and continuing...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    epoch_time_elapsed = time.monotonic() - start_time
    train_acc = 100. * correct / total if total > 0 else 0.0
    train_loss = running_loss / (len(trainloader) if len(trainloader) > 0 else 1)
    train_f1 = f1_score(all_labels, all_predictions, average='weighted') if all_labels else 0.0

    print(f'\nEpoch {epoch+1} TRAIN => Loss: {train_loss:.4f} Acc: {train_acc:.3f}% F1: {train_f1:.4f} Time: {epoch_time_elapsed:.1f}s')
    print_gpu_memory()

def test(epoch):
    global total_processing_time_test, best_Test_acc
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    avg_pool = nn.AvgPool1d(kernel_size=2)

    all_labels = []
    all_predictions = []

    start_time = time.monotonic()
    with torch.no_grad():
        for idx, (images, captions, labels) in enumerate(testloader):
            try:
                batch_start = time.time()
                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                image_features = clip_model.encode_image(images)
                image_features = image_features.unsqueeze(1)
                image_features = avg_pool(image_features).squeeze(1)

                if opt.mode == 1:
                    text_features = clip_model.encode_text(captions)
                    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
                    features = torch.cat((image_features, text_features), dim=1)
                else:
                    features = image_features

                features = nn.AvgPool1d(kernel_size=2, stride=2)(features)
                features = features.view(features.size(0), -1).float()

                outputs = net(features)
                loss = criterion(outputs, labels)

                batch_end = time.time()
                total_processing_time_test += (batch_end - batch_start)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                if idx % 50 == 0:  # Periodic memory cleanup
                    torch.cuda.empty_cache()

                utils.progress_bar(idx, len(testloader),
                                   'TestLoss: %.4f | TestAcc: %.3f%% (%d/%d)' %
                                   (running_loss / (idx + 1), 100. * correct / total, correct, total))
                                   
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM at test batch {idx}, clearing cache and continuing...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    epoch_time_elapsed = time.monotonic() - start_time
    test_acc = 100. * correct / total if total > 0 else 0.0
    test_loss = running_loss / (len(testloader) if len(testloader) > 0 else 1)
    test_f1 = f1_score(all_labels, all_predictions, average='weighted') if all_labels else 0.0

    print(f'\nEpoch {epoch+1} TEST  => Loss: {test_loss:.4f} Acc: {test_acc:.3f}% F1: {test_f1:.4f} Time: {epoch_time_elapsed:.1f}s')
    print_gpu_memory()
    
    # Early stopping
    early_stopping(test_acc)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        return True

    # Save checkpoint if improved
    if test_acc > best_Test_acc:
        print("Saving best checkpoint..")
        state = {
            'net': net.state_dict(),
            'best_Test_acc': test_acc,
            'best_Test_acc_epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, 'best_cervix_model.pth'))  # Updated checkpoint name
        best_Test_acc = test_acc

    return False

# -------------------
# Training loop with error handling
# -------------------
print(f"Starting training for {opt.epochs} epochs...")
print_gpu_memory()

total_start_time = time.monotonic()
try:
    for epoch in range(start_epoch, opt.epochs):
        print(f"\n--- Epoch {epoch+1}/{opt.epochs} ---")
        train(epoch)
        should_stop = test(epoch)
        if should_stop:
            print(f"Training stopped early at epoch {epoch+1}")
            break
        
        # Periodic full cache cleanup
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            
except KeyboardInterrupt:
    print("Training interrupted by user")
except Exception as e:
    print(f"Training error: {e}")
    raise
finally:
    total_end_time = time.monotonic()
    total_hours = int((total_end_time - total_start_time) // 3600)
    total_mins = int(((total_end_time - total_start_time) % 3600) // 60)
    total_secs = int((total_end_time - total_start_time) % 60)
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*50}")
    print(f"Total training time: {total_hours}h {total_mins}m {total_secs}s")
    print(f"Best test accuracy: {best_Test_acc:.3f}%")
    print(f"Model saved as: {os.path.join(path, 'best_cervix_model.pth')}")
    
    # Final cleanup
    if use_cuda:
        torch.cuda.empty_cache()
