import torch
import torch.nn as nn
from torchvision import models
import timm
import json
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

class ModelClassifier:
    def __init__(self, model_type: str = "resnet152"):
        """
        Args:
            model_type: "resnet152" or "vit"
        """
        if model_type not in ["resnet152", "vit"]:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # モデルとクラス名を読み込み
        self.classes = self._load_classes()
        self.model = self._load_model()
        self.model.eval()
    
    def _load_classes(self) -> List[str]:
        """クラス名を読み込み"""
        # backend/app/models/classifier.py から backend/models/ へのパス
        models_dir = Path(__file__).parent.parent.parent / "models"
        
        logger.info(f"Looking for model files in: {models_dir.absolute()}")
        
        if self.model_type == "resnet152":
            class_file = models_dir / "resnet152_classes.json"
        else:
            class_file = models_dir / "vit_classes.json"
        
        if not class_file.exists():
            raise FileNotFoundError(
                f"Class file not found: {class_file.absolute()}\n"
                f"Please ensure the model files are in the correct location."
            )
        
        logger.info(f"Loading classes from: {class_file}")
        with open(class_file, "r") as f:
            classes = json.load(f)
        
        logger.info(f"Loaded {len(classes)} classes: {classes}")
        return classes
    
    def _load_model(self) -> nn.Module:
        """学習済みモデルを読み込み"""
        models_dir = Path(__file__).parent.parent.parent / "models"
        num_classes = len(self.classes)
        
        logger.info(f"Creating model with {num_classes} classes")
        
        if self.model_type == "resnet152":
            model = models.resnet152(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            model_path = models_dir / "resnet152_best_model.pth"
        else:  # vit
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            num_features = model.head.in_features
            model.head = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            model_path = models_dir / "vit_best_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path.absolute()}\n"
                f"Please run the training script to generate the model files."
            )
        
        logger.info(f"Loading model weights from: {model_path}")
        
        # 重みを読み込み
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
        
        model.to(self.device)
        
        return model
    
    def predict(self, image_tensor: torch.Tensor, top_k: int = 5):
        """
        画像テンソルから予測を実行

        Args:
            image_tensor: 前処理済み画像テンソル (1, 3, 224, 224)
            top_k: 上位K個の予測を返す

        Returns:
            List of dicts with 'class_name' and 'confidence'
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # クラス数を超えないようにtop_kを調整
            actual_k = min(top_k, len(self.classes))

            # Top-K予測
            top_probs, top_indices = torch.topk(probabilities, actual_k)

            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                predictions.append({
                    "class_name": self.classes[idx.item()],
                    "confidence": float(prob.item())
                })

            return predictions
