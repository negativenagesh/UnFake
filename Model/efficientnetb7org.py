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

# Define data transforms with reduced resolution
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Reduced from 380x380 to save memory
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
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
            # Attempt to load pretrained weights
            self.base_model = create_model('efficientnet_b7', pretrained=True, num_classes=0)
            print("Loaded pretrained EfficientNet-B7 weights.")
        except RuntimeError as e:
            print(f"Failed to load pretrained weights: {e}. Using random initialization.")
            self.base_model = create_model('efficientnet_b7', pretrained=False, num_classes=0)
        self.fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

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

        # Clear memory after each epoch
        torch.cuda.empty_cache()

    return train_losses, val_losses, train_accs, val_accs

# Plotting function
def plot_results(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Epochs")
    plt.tight_layout()
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
    
    # Reduced batch size to prevent OOM
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Initialize model, loss, and optimizer
    model = DeepfakeClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=10
    )
    
    # Plot results
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    # Evaluate on test set
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), "efficientnet_b7_deepfake.pth")
    print("Model saved as 'efficientnet_b7_deepfake.pth'")
    
    # Clear memory after completion
    torch.cuda.empty_cache()
