import numpy as np
import onnxruntime as ort

from PIL import Image
from torchvision import transforms


# Configuration

ONNX_PATH = "fruit_classifier.onnx"
IMAGE_PATH = "test_image.jpg"

CLASS_NAMES = ["Mango", "apple", "banana", "orange"]


# Load ONNX model

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded successfully!")


# Get input name

input_name = session.get_inputs()[0].name

print("Input name:", input_name)


# Image preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load test image

image = Image.open(IMAGE_PATH).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.numpy()


# Run ONNX inference

outputs = session.run(
    None,
    {
        input_name: image
    }
)


# Get prediction

logits = outputs[0][0]

probabilities = np.exp(logits) / np.sum(np.exp(logits))

predicted_index = np.argmax(probabilities)

predicted_class = CLASS_NAMES[predicted_index]

confidence = probabilities[predicted_index] * 100


# Print result

print("\nPrediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")