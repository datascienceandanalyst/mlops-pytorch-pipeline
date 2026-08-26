import io
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import get_model


app = FastAPI(title="CIFAR-10 Model Serving API")


MODEL_PATH = Path("checkpoints/classifier_v1.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


model = None


def load_model():
    global model

    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()


try:
    load_model()
except Exception as e:
    print(f"Model loading failed: {e}")


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "healthy",
        "model": "resnet18",
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "predicted_index": predicted_index,
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]), 6)
                for i in range(len(CLASS_NAMES))
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}",
        )