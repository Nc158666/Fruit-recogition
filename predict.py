import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image


# Basic parameters
MODEL_PATH = "best_model.pth"
IMAGE_PATH = "test_image.jpg"

CLASS_NAMES = [
    "Mango",
    "apple",
    "banana",
    "orange",
]


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Create the ResNet18 model
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)


# Load the trained model
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# Load the test image
image = Image.open(IMAGE_PATH).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)


# Make a prediction
with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )


# Get the predicted class
predicted_class = CLASS_NAMES[predicted.item()]

confidence = confidence.item() * 100


print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")