# -*- coding: utf-8 -*-
"""FER System - Emotion Predictor

Model inference wrapper for facial expression prediction.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.student_model import ImprovedMobileNetV3Small
from models.student_model_config import NUM_CLASSES


EMOTION_LABELS: list = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]

NORMALIZE_MEAN: list = [0.5, 0.5, 0.5]
NORMALIZE_STD: list = [0.5, 0.5, 0.5]


class EmotionPredictor:
    """Facial expression prediction using a trained student model.

    Loads the improved MobileNetV3-Small student model and provides
    single-image and batch prediction capabilities.

    Args:
        model_path: Path to the student model weights (.pth).
        device: Inference device ('cpu' or 'cuda').
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # Load model
        self.model = ImprovedMobileNetV3Small(num_classes=NUM_CLASSES)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])

    @torch.no_grad()
    def predict(self, face_image: np.ndarray) -> Dict[str, float]:
        """Predict emotion probabilities for a single face image.

        Args:
            face_image: RGB face image [H, W, 3], pixel range [0, 255].

        Returns:
            Dictionary mapping emotion names to probabilities.
        """
        # Convert BGR to RGB if needed
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
        elif face_image.shape[2] == 3:
            # Assume BGR from OpenCV
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Preprocess
        face_resized = cv2.resize(face_image, (48, 48))
        tensor = self.transform(face_resized).unsqueeze(0).to(self.device)

        # Inference
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        return {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}

    @torch.no_grad()
    def predict_batch(self, face_images: List[np.ndarray]) -> List[Dict[str, float]]:
        """Predict emotions for a batch of face images.

        Args:
            face_images: List of RGB face images.

        Returns:
            List of emotion probability dictionaries.
        """
        tensors = []
        for img in face_images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (48, 48))
            tensors.append(self.transform(img_resized))

        batch_tensor = torch.stack(tensors).to(self.device)
        logits = self.model(batch_tensor)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for prob_row in probs:
            results.append(
                {label: float(p) for label, p in zip(EMOTION_LABELS, prob_row)}
            )
        return results

    @torch.no_grad()
    def predict_topk(
        self, face_image: np.ndarray, k: int = 3
    ) -> List[Tuple[str, float]]:
        """Get top-K emotion predictions.

        Args:
            face_image: RGB face image.
            k: Number of top predictions.

        Returns:
            List of (emotion, probability) tuples sorted descending.
        """
        probs = self.predict(face_image)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]
