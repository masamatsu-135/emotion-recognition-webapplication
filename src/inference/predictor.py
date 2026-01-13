from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import torch

from ..training.model import get_model
from ..training.dataset import get_resnet_transforms

from .labels import EMOTION_LABELS_EN, EMOTION_LABELS_JA, get_label_en, get_label_ja


ModelType = ["resnet"]

class EmotionPredictor:
    def __init__(
        self,
        model_type = "resnet",
        checkpoint_path = "models/checkpoints/best_resnet_fer2013.pth",
        device = None
    ):
        self.model_type = model_type

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.transform = get_resnet_transforms(train=False)

        self.model = get_model(
            model_type=model_type,
            num_classes=7,
            pretrained=False,
            device=self.device
        )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
            )
        
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[INFO] Loaded model_type='{model_type}' from: {checkpoint_path}")
        print(f"[INFO] Using device: {self.device}")


    def to_pil(self, image, bgr=False):
        
        if isinstance(image, Image.Image):
            return image
        
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"Expected image shape (H, W, 3), got {image.shape}"
                )
            
            img = image
            if bgr:
                img = img[:, :, ::-1]

            return Image.fromarray(img.astype(np.uint8))
        
        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Use PIL.Image.Image or np.ndarray."
        )
    
    def preprocess(self, image, bgr=False):
        pil_img = self.to_pil(image, bgr=bgr)
        tensor = self.transform(pil_img)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)
    

    def predict_from_pil(self, image):
        input_tensor = self.preprocess(image, bgr=False)
        return self.predict_from_tensor(input_tensor)
    
    def predict_from_ndarray(self, image, bgr=False):
        input_tensor = self.preprocess(image, bgr=bgr)
        return self.predict_from_tensor(input_tensor)
    
    def predict_from_tensor(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]

        class_id = int(torch.argmax(probs).item())
        probs_list = probs.detach().cpu().numpy().tolist()

        result = {
            "class_id": class_id,
            "label_en": get_label_en(class_id),
            "label_ja": get_label_ja(class_id),
            "probs": probs_list,
        }
        return result
    

if __name__ == "__main__":
    import os

    DEFAULT_CHECKPOINT = Path("models/checkpoints/best_resnet_fer2013.pth")
    image_path = Path("tests/face.jpg")

    if image_path is None or not Path(image_path).exists():
        print("[WARN] テスト用の顔画像パスを predictor.py 内で設定してください。")
    else:
        predictor = EmotionPredictor(
            model_type="resnet",
            checkpoint_path=DEFAULT_CHECKPOINT,
            device=None,
        )
        img = Image.open(image_path).convert("RGB")
        result = predictor.predict_from_pil(img)
        print("Prediction result:", result)
