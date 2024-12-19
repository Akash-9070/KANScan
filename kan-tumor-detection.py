import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class KolmogorovLayer(nn.Module):
    def __init__(self, input_dim, inner_dim):
        super(KolmogorovLayer, self).__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        
        # Initialize Ψ functions (inner functions)
        self.psi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, inner_dim),
                nn.Tanh(),
                nn.Linear(inner_dim, inner_dim)
            ) for _ in range(input_dim)
        ])
        
        # Initialize g function (outer function)
        self.g = nn.Sequential(
            nn.Linear(input_dim * inner_dim, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply Ψ functions to each dimension
        psi_outputs = []
        for i in range(self.input_dim):
            xi = x[:, i].view(-1, 1)
            psi_i = self.psi[i](xi)
            psi_outputs.append(psi_i)
        
        # Concatenate all Ψ outputs
        psi_concat = torch.cat(psi_outputs, dim=1)
        
        # Apply g function
        out = self.g(psi_concat)
        return out

class KANetwork(nn.Module):
    def __init__(self, input_channels=1, hidden_dims=[64, 128, 256]):
        super(KANetwork, self).__init__()
        
        # Feature extraction using convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate feature dimension after convolutions
        self.feature_dim = hidden_dims[2] * (256 // 8) * (256 // 8)  # Assuming 256x256 input
        
        # Kolmogorov-Arnold representation
        self.kan_layers = nn.ModuleList([
            KolmogorovLayer(input_dim=self.feature_dim, inner_dim=64),
            KolmogorovLayer(input_dim=64, inner_dim=32),
            KolmogorovLayer(input_dim=32, inner_dim=16)
        ])
        
        # Bbox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # (x, y, width, height)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Apply KAN layers
        kan_out = features
        for kan_layer in self.kan_layers:
            kan_out = kan_layer(kan_out)
            
        # Get predictions
        bbox = self.bbox_head(kan_out)
        cls_prob = self.cls_head(kan_out)
        
        return bbox, cls_prob

class CTScanDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.annotations = []
        
        # Load annotations (image_name,x,y,width,height,label)
        with open(annotation_file, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                self.annotations.append({
                    'image': data[0],
                    'bbox': [float(x) for x in data[1:5]],
                    'label': int(data[5])
                })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.image_dir, ann['image'])
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'bbox': torch.tensor(ann['bbox'], dtype=torch.float32),
            'label': torch.tensor([ann['label']], dtype=torch.float32)
        }

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion_bbox = nn.SmoothL1Loss()
    criterion_cls = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            bbox_targets = batch['bbox'].to(device)
            cls_targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            bbox_pred, cls_pred = model(images)
            
            loss_bbox = criterion_bbox(bbox_pred, bbox_targets)
            loss_cls = criterion_cls(cls_pred, cls_targets)
            loss = loss_bbox + loss_cls
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                bbox_targets = batch['bbox'].to(device)
                cls_targets = batch['label'].to(device)
                
                bbox_pred, cls_pred = model(images)
                loss_bbox = criterion_bbox(bbox_pred, bbox_targets)
                loss_cls = criterion_cls(cls_pred, cls_targets)
                val_loss += (loss_bbox + loss_cls).item()
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

def detect_tumor(model, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    original_size = image.size
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        bbox_pred, cls_pred = model(image_tensor)
    
    # Convert bbox to original image size
    bbox = bbox_pred[0].cpu().numpy()
    bbox[0] *= original_size[0]  # x
    bbox[1] *= original_size[1]  # y
    bbox[2] *= original_size[0]  # width
    bbox[3] *= original_size[1]  # height
    
    confidence = cls_pred[0].item()
    
    # Visualize results
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    
    if confidence > 0.5:
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0], bbox[1] - 10,
            f'Tumor: {confidence:.2f}',
            color='red'
        )
    
    plt.axis('off')
    plt.show()
    
    return bbox, confidence

def main():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = CTScanDataset(
        image_dir='path/to/train/images',
        annotation_file='path/to/train/annotations.txt',
        transform=transform
    )
    
    val_dataset = CTScanDataset(
        image_dir='path/to/val/images',
        annotation_file='path/to/val/annotations.txt',
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = KANetwork()
    
    # Train model
    train_model(model, train_loader, val_loader)
    
    # Save trained model
    torch.save(model.state_dict(), 'tumor_detection_kan.pth')
    
    # Test on single image
    test_image_path = 'path/to/test/image.jpg'
    bbox, confidence = detect_tumor(model, test_image_path)
    print(f'Detection Results:')
    print(f'Bounding Box: {bbox}')
    print(f'Confidence: {confidence:.2f}')

if __name__ == '__main__':
    main()