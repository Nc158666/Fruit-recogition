import torch
import torch.nn as nn
from torchvision import models


# Configuration

MODEL_PATH = "best_model.pth"
ONNX_PATH = "fruit_classifier.onnx"

CLASS_NAMES = ["Mango", "apple", "banana", "orange"]


# Select device

device = torch.device("cpu")


# Load model

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

model.to(device)


# Create dummy input

dummy_input = torch.randn(
    1, 3, 224, 224
).to(device)


# Export model to ONNX

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {
            0: "batch_size"
        },
        "output": {
            0: "batch_size"
        }
    }
)


print("ONNX model exported successfully!")
print(f"Model saved as: {ONNX_PATH}")