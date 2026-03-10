"""
main.py
-------
FastAPI web application for Rice Leaf Disease Classification.

Features:
    - Upload a rice leaf image → get disease prediction + confidence
    - Returns top-3 predictions with probabilities
    - Grad-CAM explainability (shows what the model looked at)
    - Health check endpoint
    - Free to use by anyone, no authentication required

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open: http://localhost:8000
"""

import io
import os
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.predictor import RiceLeafPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# App Initialization
# =============================================================================

app = FastAPI(
    title="Rice Leaf Disease Classifier",
    description=(
        "AI-powered rice leaf disease detection using VGG16 deep learning. "
        "Upload a rice leaf image to identify: Bacterial Blight, Blast, Brown Spot, or Tungro."
    ),
    version="1.0.0",
)

# Allow requests from any origin (free public access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, uploaded images)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load the trained model once at startup
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load model when server starts."""
    global predictor
    model_path = os.getenv("MODEL_PATH", "outputs/models/vgg16/final_vgg16.keras")
    logger.info(f"Loading model from: {model_path}")
    predictor = RiceLeafPredictor(model_path=model_path)
    logger.info("Model loaded successfully. Server ready.")


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    html_path = Path("app/templates/index.html")
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Rice Leaf Disease Classifier API</h1><p>Visit /docs for API documentation.</p>"


@app.get("/health")
async def health_check():
    """Simple health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": time.time(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict rice leaf disease from an uploaded image.

    Accepts: JPG, JPEG, PNG, WEBP images.
    Returns: Disease class, confidence, top-3 predictions.

    Example response:
    {
        "predicted_class": "Bacterial Blight",
        "confidence": 0.9876,
        "top_predictions": [
            {"class": "Bacterial Blight", "probability": 0.9876},
            {"class": "Blast",            "probability": 0.0087},
            {"class": "Brown Spot",       "probability": 0.0023}
        ],
        "inference_time_ms": 45.2
    }
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a JPG, PNG, or WebP image."
        )

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again shortly.")

    # Read and process image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    # Run prediction
    start_time = time.time()
    result = predictor.predict(image)
    inference_time_ms = (time.time() - start_time) * 1000

    return {
        "predicted_class":  result["predicted_class"],
        "confidence":        round(result["confidence"], 4),
        "top_predictions":   result["top_predictions"],
        "inference_time_ms": round(inference_time_ms, 2),
    }


@app.post("/predict-with-gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Predict disease AND generate a Grad-CAM explainability image.

    Returns the same as /predict plus a URL to the Grad-CAM heatmap overlay.
    This shows WHICH areas of the leaf the model focused on.
    """
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")

    start_time = time.time()
    result = predictor.predict(image)
    gradcam_path = predictor.generate_gradcam(image, result["predicted_class_idx"])
    inference_time_ms = (time.time() - start_time) * 1000

    return {
        "predicted_class":  result["predicted_class"],
        "confidence":        round(result["confidence"], 4),
        "top_predictions":   result["top_predictions"],
        "gradcam_image_url": f"/static/uploads/{Path(gradcam_path).name}",
        "inference_time_ms": round(inference_time_ms, 2),
    }


@app.get("/classes")
async def get_classes():
    """Return the list of disease classes the model can detect."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "classes": predictor.class_names,
        "count": len(predictor.class_names),
        "descriptions": {
            "Bacterialblight": "Caused by Xanthomonas oryzae. Symptoms: yellowing and wilting of leaves.",
            "Blast":           "Caused by Magnaporthe oryzae. Symptoms: diamond-shaped lesions on leaves.",
            "Brownspot":       "Caused by Bipolaris oryzae. Symptoms: small brown circular lesions.",
            "Tungro":          "Viral disease. Symptoms: yellow-orange discoloration and stunted growth.",
        }
    }


# =============================================================================
# Run directly
# =============================================================================

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
