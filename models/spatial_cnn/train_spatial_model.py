import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================
# CONFIG
# ==========================
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_SAVE_PATH = "models/saved_models/spatial_model.pth"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing Device: {DEVICE}")

# ==========================
# DATA AUGMENTATION
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# LOAD DATASETS
# ==========================
train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nClasses Found:", train_dataset.classes)

if len(train_dataset.classes) != 3:
    raise ValueError(
        f"Expected exactly 3 classes: ['fake', 'real', 'spoof'] or equivalent. "
        f"Found: {train_dataset.classes}"
    )

# ==========================
# BUILD MODEL
# ==========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze feature extractor first for more stable training on small datasets
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

# Train classifier head
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ==========================
# LOSS + OPTIMIZER
# ==========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LR
)

# ==========================
# TRAINING LOOP
# ==========================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\n==============================")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"==============================")

    # --------------------------
    # TRAINING
    # --------------------------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100.0 * correct / total
    avg_train_loss = running_loss / len(train_loader)

    # --------------------------
    # VALIDATION
    # --------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100.0 * val_correct / val_total

    # --------------------------
    # PRINT RESULTS
    # --------------------------
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # --------------------------
    # SAVE BEST MODEL
    # --------------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc

        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": train_dataset.classes
            },
            MODEL_SAVE_PATH
        )

        print("Best Model Saved!")

print("\n==============================")
print("TRAINING COMPLETE")
print(f"BEST VALIDATION ACCURACY: {best_val_acc:.2f}%")
print("==============================")