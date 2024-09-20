import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = (128, 128)  # Resize images to this size (adjust as needed)
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None):
        self.data = pd.read_csv(annotations_file)
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])  # Get the image path
        image = Image.open(img_name).convert("RGB")  # Open the image
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Get the coordinates (x7, y7, x8, y8)
        coordinates = self.data.iloc[idx, 1:].values.astype(float)
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        
        return image, coordinates

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjust based on image size after pooling
        self.fc2 = nn.Linear(128, 4)  # Output for x7, y7, x8, y8
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset and apply transformations
def load_data(annotation_file, image_dir):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(annotation_file, image_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# Train the model
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "coordinate_predictor.pth")

# Example usage
annotation_file = 'p14_normalized_file.csv'  # Your CSV file with normalized coordinates
image_dir = 'path/to/image/directory'  # Path to directory containing images

# Load data
train_loader, val_loader = load_data(annotation_file, image_dir)

# Initialize and train the model
model = CNNModel().to(DEVICE)
train_model(model, train_loader, val_loader)

print("Model training complete!")
