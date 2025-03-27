import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np
import seaborn as sns

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear GPU memory before starting
torch.cuda.empty_cache()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define data transforms with improved augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Increased rotation range
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Enhanced color jitter
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added scaling
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Added perspective changes
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # Random erasing for robustness
    ]),
    'val': transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load dataset with separate real and fake directories
def load_dataset(real_dir, fake_dir):
    if not os.path.isdir(real_dir):
        raise ValueError(f"Directory not found: {real_dir}")
    if not os.path.isdir(fake_dir):
        raise ValueError(f"Directory not found: {fake_dir}")
    
    real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, img))]
    fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, img))]
    
    if not real_images:
        raise ValueError(f"No images found in {real_dir}")
    if not fake_images:
        raise ValueError(f"No images found in {fake_dir}")
    
    image_paths = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

# Define the classifier model
class DeepfakeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeClassifier, self).__init__()
        try:
            # Load pretrained weights
            self.base_model = create_model('efficientnet_b7', pretrained=True, num_classes=0)
            print("Loaded pretrained EfficientNet-B7 weights.")
        except RuntimeError as e:
            print(f"Failed to load pretrained weights: {e}. Using random initialization.")
            self.base_model = create_model('efficientnet_b7', pretrained=False, num_classes=0)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Comprehensive evaluation metrics
def evaluate_model(model, data_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    loss = running_loss / len(data_loader)
    accuracy = np.mean(all_preds == all_labels) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_probs)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }
    
    return metrics

# Training function with early stopping and model checkpointing
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10, patience=3):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate scheduler if provided
        if scheduler:
            scheduler.step(val_loss)
            
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, "best_model_checkpoint.pth")
            print(f"Checkpoint saved (val_loss: {val_loss:.4f})")
        else:
            early_stopping_counter += 1
            
        # Early stopping
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Clear memory after each epoch
        torch.cuda.empty_cache()
    
    # Load the best model before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return train_losses, val_losses, train_accs, val_accs

# Plotting function
def plot_results(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label="Train Loss")
    plt.plot(epochs, val_losses, 'r-', label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label="Train Acc")
    plt.plot(epochs, val_accs, 'r-', label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Epochs")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Define dataset paths
    real_dir = '/home/vu-lab03-pc24/Downloads/Real'
    fake_dir = '/home/vu-lab03-pc24/Downloads/fake'
    
    # Load and prepare data
    try:
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = load_dataset(real_dir, fake_dir)
        print(f"Number of training images: {len(train_paths)}")
        print(f"Number of validation images: {len(val_paths)}")
        print(f"Number of test images: {len(test_paths)}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    train_dataset = DeepfakeDataset(train_paths, train_labels, data_transforms['train'])
    val_dataset = DeepfakeDataset(val_paths, val_labels, data_transforms['val'])
    test_dataset = DeepfakeDataset(test_paths, test_labels, data_transforms['val'])
    
    # Use smaller batch size for large images
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model, loss, and optimizer
    model = DeepfakeClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Train the model with early stopping and scheduler
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler=scheduler, num_epochs=15, patience=5
    )
    
    # Plot results
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    # Comprehensive evaluation on test set
    test_metrics = evaluate_model(model, test_loader, criterion)
    
    print(f"\nTest Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Save the model
    torch.save(model.state_dict(), "efficientnet_b7_deepfake.pth")
    print("Model saved as 'efficientnet_b7_deepfake.pth'")
    
    # Clear memory after completion
    torch.cuda.empty_cache()