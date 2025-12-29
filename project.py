# %% [markdown]
# # Deepfake Detection System
# ## Step-by-Step Implementation
# 
# This notebook implements a complete deepfake detection system using deep learning.
# 
# ---
# 
# ### Dataset Structure Required:
# ```
# your_project_folder/
# ├── Celeb-real/          # Real celebrity videos
# │   ├── video1.mp4
# │   ├── video2.mp4
# │   └── ...
# ├── Celeb-synthesis/     # Fake/deepfake videos
# │   ├── video1.mp4
# │   ├── video2.mp4
# │   └── ...
# ├── YouTube-real/        # Real YouTube videos
# │   ├── video1.mp4
# │   ├── video2.mp4
# │   └── ...
# └── deepfake_detection.ipynb  # This notebook
# ```
# 
# **IMPORTANT:** Place this notebook in the same folder where your video folders are located!

# %% [markdown]
# ## Step 1: Install Required Libraries
# Run this cell first to install all dependencies

# %%
# !pip install torch torchvision opencv-python numpy scikit-learn matplotlib seaborn tqdm pillow
print("✓ All libraries installed successfully!")

# %% [markdown]
# ## Step 2: Import Libraries

# %%
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully!")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# %% [markdown]
# ## Step 3: Set Configuration Parameters

# %%
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration parameters
CONFIG = {
    'base_path': '.',                    # Current directory (where notebook is located)
    'frame_extraction_rate': 10,         # Extract 1 frame every 10 frames
    'max_frames_per_video': 20,          # Maximum frames per video
    'max_videos_per_folder': 50,         # Limit videos for faster processing (set to None for all)
    'img_size': 224,                     # Image size for model input
    'batch_size': 16,                    # Batch size for training
    'epochs': 10,                        # Number of training epochs
    'learning_rate': 0.0001,             # Learning rate
}

print("\n" + "="*50)
print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print("="*50)

# %% [markdown]
# ## Step 4: Verify Dataset Structure
# This cell checks if your dataset folders exist

# %%
def check_dataset_structure(base_path):
    """Check if dataset folders exist"""
    folders = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
    print("Checking dataset structure...\n")
    
    all_exist = True
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        exists = os.path.exists(folder_path)
        
        if exists:
            video_files = [f for f in os.listdir(folder_path) 
                          if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
            print(f"✓ {folder:20s} - Found {len(video_files)} videos")
        else:
            print(f"✗ {folder:20s} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✓ All dataset folders found!")
    else:
        print("\n⚠ Some folders are missing. Please check your dataset structure.")
        print("\nExpected structure:")
        print("  your_folder/")
        print("  ├── Celeb-real/")
        print("  ├── Celeb-synthesis/")
        print("  ├── YouTube-real/")
        print("  └── this_notebook.ipynb")
    
    return all_exist

dataset_ok = check_dataset_structure(CONFIG['base_path'])

# %% [markdown]
# ## Step 5: Frame Extraction Functions
# Extract frames from videos for analysis

# %%
def extract_frames_from_video(video_path, max_frames=20, frame_skip=10):
    """Extract frames from a video file"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return frames
    
    frame_count = 0
    extracted = 0
    
    while cap.isOpened() and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (CONFIG['img_size'], CONFIG['img_size']))
            frames.append(frame)
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    return frames

def load_dataset(base_path, max_videos=None):
    """Load and prepare dataset"""
    data = []
    labels = []
    
    # Real videos (Celeb-real and YouTube-real)
    real_folders = ['Celeb-real', 'YouTube-real']
    for folder in real_folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            print(f"\n{'='*60}")
            print(f"Processing {folder}...")
            print('='*60)
            
            video_files = [f for f in os.listdir(folder_path) 
                          if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
            
            if max_videos:
                video_files = video_files[:max_videos]
            
            for video_file in tqdm(video_files, desc=f"Loading {folder}"):
                video_path = os.path.join(folder_path, video_file)
                frames = extract_frames_from_video(video_path, 
                                                   CONFIG['max_frames_per_video'], 
                                                   CONFIG['frame_extraction_rate'])
                
                for frame in frames:
                    data.append(frame)
                    labels.append(0)  # 0 for real
            
            print(f"✓ Loaded {len(video_files)} videos from {folder}")
    
    # Fake videos (Celeb-synthesis)
    fake_folder = os.path.join(base_path, 'Celeb-synthesis')
    if os.path.exists(fake_folder):
        print(f"\n{'='*60}")
        print(f"Processing Celeb-synthesis...")
        print('='*60)
        
        video_files = [f for f in os.listdir(fake_folder) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        for video_file in tqdm(video_files, desc="Loading Celeb-synthesis"):
            video_path = os.path.join(fake_folder, video_file)
            frames = extract_frames_from_video(video_path, 
                                               CONFIG['max_frames_per_video'], 
                                               CONFIG['frame_extraction_rate'])
            
            for frame in frames:
                data.append(frame)
                labels.append(1)  # 1 for fake
        
        print(f"✓ Loaded {len(video_files)} videos from Celeb-synthesis")
    
    return np.array(data), np.array(labels)

print("✓ Frame extraction functions defined!")

# %% [markdown]
# ## Step 6: Load and Prepare Dataset
# This will take a few minutes depending on the number of videos

# %%
print("Loading dataset... This may take a few minutes.\n")

X, y = load_dataset(CONFIG['base_path'], CONFIG['max_videos_per_folder'])

print(f"\n{'='*60}")
print("Dataset Summary:")
print('='*60)
print(f"Total samples: {len(X)}")
print(f"Real samples: {np.sum(y == 0)}")
print(f"Fake samples: {np.sum(y == 1)}")
print(f"Image shape: {X[0].shape}")
print('='*60)

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Sample Images from Dataset', fontsize=16)

for i in range(5):
    # Real images
    real_idx = np.where(y == 0)[0][i]
    axes[0, i].imshow(X[real_idx])
    axes[0, i].set_title('Real')
    axes[0, i].axis('off')
    
    # Fake images
    fake_idx = np.where(y == 1)[0][i]
    axes[1, i].imshow(X[fake_idx])
    axes[1, i].set_title('Fake')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7: Split Dataset into Train, Validation, and Test Sets

# %%
# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Dataset Split:")
print(f"  Training set:   {len(X_train)} samples")
print(f"  Validation set: {len(X_val)} samples")
print(f"  Test set:       {len(X_test)} samples")

# %% [markdown]
# ## Step 8: Create Custom Dataset Class

# %%
class DeepfakeDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

print("✓ Dataset class created!")

# %% [markdown]
# ## Step 9: Define Data Transformations

# %%
# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test/validation transforms without augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("✓ Data transformations defined!")

# %% [markdown]
# ## Step 10: Create DataLoaders

# %%
# Create datasets
train_dataset = DeepfakeDataset(X_train, y_train, train_transform)
val_dataset = DeepfakeDataset(X_val, y_val, test_transform)
test_dataset = DeepfakeDataset(X_test, y_test, test_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print("✓ DataLoaders created!")
print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# %% [markdown]
# ## Step 11: Define the Model Architecture

# %%
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Use pre-trained EfficientNet-B0
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.model(x)

print("✓ Model architecture defined!")

# %% [markdown]
# ## Step 12: Initialize Model, Loss Function, and Optimizer

# %%
# Initialize model
model = DeepfakeDetector().to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model Summary:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# %% [markdown]
# ## Step 13: Define Training Function

# %%
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{running_loss/len(progress_bar):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

print("✓ Training functions defined!")

# %% [markdown]
# ## Step 14: Train the Model
# This is the main training loop

# %%
print("Starting training...\n")

train_losses = []
val_losses = []
train_accs = []
val_accs = []

best_val_acc = 0.0

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    print("-" * 60)
    
    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")

print("\n" + "="*60)
print("Training completed!")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print("="*60)

# %% [markdown]
# ## Step 15: Plot Training History

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss plot
ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(val_losses, label='Val Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(train_accs, label='Train Accuracy', marker='o')
ax2.plot(val_accs, label='Val Accuracy', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Training history plot saved as 'training_history.png'")

# %% [markdown]
# ## Step 16: Evaluate on Test Set

# %%
print("Evaluating on test set...\n")

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate accuracy
test_accuracy = accuracy_score(all_labels, all_preds)

print("\n" + "="*60)
print(f"TEST ACCURACY: {test_accuracy * 100:.2f}%")
print("="*60)

# %% [markdown]
# ## Step 17: Detailed Classification Report

# %%
print("\nDetailed Classification Report:")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], digits=4))

# %% [markdown]
# ## Step 18: Confusion Matrix Visualization

# %%
# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], 
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add percentage annotations
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm[i].sum() * 100
        plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# %% [markdown]
# ## Step 19: Sample Predictions Visualization

# %%
# Get some sample predictions
model.eval()
num_samples = 10

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Sample Predictions', fontsize=16)

sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

for idx, sample_idx in enumerate(sample_indices):
    row = idx // 5
    col = idx % 5
    
    # Get image and true label
    image = X_test[sample_idx]
    true_label = y_test[sample_idx]
    
    # Prepare image for model
    image_tensor = test_transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        pred_label = predicted.item()
    
    # Plot
    axes[row, col].imshow(image)
    title_color = 'green' if pred_label == true_label else 'red'
    title = f"True: {'Fake' if true_label else 'Real'}\nPred: {'Fake' if pred_label else 'Real'}"
    axes[row, col].set_title(title, color=title_color, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Sample predictions saved as 'sample_predictions.png'")

# %% [markdown]
# ## Step 20: Save Final Model

# %%
# Save final model
torch.save(model.state_dict(), 'deepfake_detector_final.pth')
print("✓ Final model saved as 'deepfake_detector_final.pth'")

# Save configuration
import json
with open('model_config.json', 'w') as f:
    json.dump(CONFIG, f, indent=4)
print("✓ Configuration saved as 'model_config.json'")

# %% [markdown]
# ## Summary and Results
# 
# ### Files Generated:
# 1. **best_model.pth** - Best model weights (highest validation accuracy)
# 2. **deepfake_detector_final.pth** - Final model weights
# 3. **training_history.png** - Training curves
# 4. **confusion_matrix.png** - Confusion matrix visualization
# 5. **sample_predictions.png** - Sample prediction results
# 6. **model_config.json** - Model configuration
# 
# ### Key Metrics:

# %%
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Total Training Time: {CONFIG['epochs']} epochs")
print(f"Model: EfficientNet-B0 (Transfer Learning)")
print(f"Dataset: {len(X)} total samples")
print(f"  - Real samples: {np.sum(y == 0)}")
print(f"  - Fake samples: {np.sum(y == 1)}")
print("="*70)

# %% [markdown]
# ## How to Use This Model for Predictions
# 
# Use the following code to make predictions on new videos:

# %%
def predict_video(video_path, model, transform, device):
    """
    Predict if a video is real or fake
    
    Args:
        video_path: Path to video file
        model: Trained model
        transform: Image transformation
        device: torch device
    
    Returns:
        prediction: 0 for real, 1 for fake
        confidence: Prediction confidence (0-1)
    """
    # Extract frames
    frames = extract_frames_from_video(video_path, max_frames=20, frame_skip=10)
    
    if len(frames) == 0:
        return None, 0.0
    
    # Make predictions on all frames
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for frame in frames:
            image_tensor = transform(frame).unsqueeze(0).to(device)
            output = model(image_tensor)
            prob = torch.softmax(output, dim=1)
            predictions.append(prob.cpu().numpy()[0])
    
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    final_prediction = np.argmax(avg_pred)
    confidence = avg_pred[final_prediction]
    
    return final_prediction, confidence

# Example usage:
# prediction, confidence = predict_video('path/to/video.mp4', model, test_transform, device)
# print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
# print(f"Confidence: {confidence:.2%}")

print("✓ Prediction function defined!")
print("\nYou can now use predict_video() function to classify new videos!")

# %% [markdown]
# ## 🎉 Congratulations!
# 
# You have successfully:
# - ✅ Loaded and preprocessed video data
# - ✅ Built a deepfake detection model
# - ✅ Trained the model with transfer learning
# - ✅ Evaluated the model performance
# - ✅ Generated visualizations and reports
# 
# ### Next Steps:
# 1. Try adjusting hyperparameters (epochs, learning rate, batch size)
# 2. Experiment with different models (ResNet, Xception)
# 3. Add more data augmentation techniques
# 4. Implement cross-validation
# 5. Deploy the model as a web application