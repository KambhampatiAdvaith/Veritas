import torch
import torch.nn as nn
import torch.optim as optim
import timm
import os
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --- CONFIGURATION ---
# IMPORTANT: Point this to the 'dataset' folder you created earlier
# If your dataset folder is outside this project, put the full path here.
# Example: r"C:\Users\Advaith\Desktop\dataset"
DATA_DIR = 'dataset'  
MODEL_SAVE_PATH = 'veritas_production_model.pth'

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    print(f"🚀 Starting Veritas Vision Training on {DEVICE}...")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset folder '{DATA_DIR}' not found.")
        print("   Please move your 'dataset' folder (with real/fake images) into this directory.")
        return

    # 1. Data Transformations (Augmentation for Robustness)
    train_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # 2. Load Data
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    
    # 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    print(f"📂 Training on {len(train_data)} images | Validating on {len(val_data)} images")
    print(f"ℹ️ Classes: {full_dataset.class_to_idx}")

    # 3. Load XceptionNet (Pre-trained)
    model = timm.create_model('legacy_xception', pretrained=True, num_classes=1)
    model = model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).float()
                outputs = model(images).squeeze(1)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = 100 * correct / total
        print(f"📈 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    # 5. Save the Best Model
    torch.save(best_weights, MODEL_SAVE_PATH)
    print(f"\n✅ Model Saved: {MODEL_SAVE_PATH} (Accuracy: {best_acc:.2f}%)")
    print("👉 Now you can run 'streamlit run app.py'")

if __name__ == "__main__":
    train_model()