import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# Configuration

DATASET_PATH = "fruit_dataset"

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001

# Set random seed for reproducibility
torch.manual_seed(42)


# Data preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),

    # ImageNet normalization for ResNet18
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load dataset

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# Split dataset

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Create ResNet18 model

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Replace the final classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    len(dataset.classes)
)


# Select device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = model.to(device)


# Loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training

best_val_accuracy = 0.0

for epoch in range(NUM_EPOCHS):

    # Training phase

    model.train()

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total


    # Validation phase

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total


    # Print results

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


    # Save the best model

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved.")


# Training finished

print("\nTraining finished!")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print("Model saved as: best_model.pth")