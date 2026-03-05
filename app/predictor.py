"""
predictor.py
------------
The RiceLeafPredictor class handles all model inference.

Separating inference logic from the API keeps code clean and testable.
The predictor is loaded ONCE at startup and reused for all requests.
"""

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Class names — must match the order used during training
DEFAULT_CLASS_NAMES = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]

# Human-readable display names
DISPLAY_NAMES = {
    "Bacterialblight": "Bacterial Blight",
    "Blast":           "Blast",
    "Brownspot":       "Brown Spot",
    "Tungro":          "Tungro",
}


class RiceLeafPredictor:
    """
    Loads a trained Keras model and provides prediction + Grad-CAM methods.

    Usage:
        predictor = RiceLeafPredictor("outputs/models/vgg16/final_vgg16.keras")
        result = predictor.predict(pil_image)
    """

    def __init__(self, model_path: str, class_names: list = None, image_size: tuple = (224, 224)):
        """
        Args:
            model_path:   Path to the saved .keras model file.
            class_names:  List of class names. Defaults to 4 rice disease classes.
            image_size:   Input image dimensions (height, width). Default (224, 224).
        """
        self.model_path   = model_path
        self.class_names  = class_names or DEFAULT_CLASS_NAMES
        self.image_size   = image_size
        self.model        = self._load_model()
        self.gradcam_dir  = Path("app/static/uploads")
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> tf.keras.Model:
        """Load and return the Keras model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. "
                "Please train the model first: python src/models/train.py --model vgg16"
            )
        logger.info(f"Loading model: {self.model_path}")
        model = tf.keras.models.load_model(self.model_path)
        logger.info("Model loaded successfully.")
        return model

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Resize and normalize a PIL image for model input.

        Args:
            image: PIL Image (RGB).

        Returns:
            np.ndarray of shape (1, H, W, 3) with values in [0, 1].
        """
        image = image.resize(self.image_size)
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)  # Add batch dimension

    def predict(self, image: Image.Image) -> dict:
        """
        Run prediction on a PIL image.

        Args:
            image: PIL Image (RGB).

        Returns:
            dict with keys:
                - predicted_class      (str)  — top predicted class display name
                - predicted_class_idx  (int)  — index of predicted class
                - confidence           (float) — probability of top class
                - top_predictions      (list)  — top-3 class/probability pairs
        """
        img_array   = self.preprocess(image)
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Sort classes by probability (highest first)
        ranked_indices = np.argsort(predictions)[::-1]

        top_class_idx  = int(ranked_indices[0])
        top_class_name = self.class_names[top_class_idx]
        confidence     = float(predictions[top_class_idx])

        top_predictions = [
            {
                "class":       DISPLAY_NAMES.get(self.class_names[i], self.class_names[i]),
                "raw_class":   self.class_names[i],
                "probability": round(float(predictions[i]), 4),
            }
            for i in ranked_indices[:3]
        ]

        return {
            "predicted_class":     DISPLAY_NAMES.get(top_class_name, top_class_name),
            "predicted_class_idx": top_class_idx,
            "confidence":          confidence,
            "top_predictions":     top_predictions,
        }

    def generate_gradcam(self, image: Image.Image, class_idx: int) -> str:
        """
        Generate a Grad-CAM overlay image and save it.

        Args:
            image:     PIL Image (RGB).
            class_idx: The class index to generate Grad-CAM for.

        Returns:
            str: Path to the saved Grad-CAM image.
        """
        img_array = self.preprocess(image)

        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            logger.warning("No convolutional layer found for Grad-CAM.")
            return None

        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(last_conv_layer).output,
                self.model.output,
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize and colorize heatmap
        h, w = self.image_size
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        original = np.array(image.resize(self.image_size), dtype=np.uint8)
        superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

        # Save with a unique filename
        filename = f"gradcam_{uuid.uuid4().hex[:8]}.jpg"
        save_path = self.gradcam_dir / filename
        Image.fromarray(superimposed).save(str(save_path))

        return str(save_path)
