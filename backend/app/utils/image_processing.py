from PIL import Image
import torch
from torchvision import transforms
import io

def preprocess_image(image_bytes: bytes, model_type: str) -> torch.Tensor:
    """
    画像バイトデータを前処理してテンソルに変換
    
    Args:
        image_bytes: 画像のバイトデータ
        model_type: "resnet152" or "vit"
    
    Returns:
        前処理済みテンソル (1, 3, 224, 224)
    """
    # 画像を開く
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # モデルに応じた前処理
    if model_type == "resnet152":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # vit
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    # テンソルに変換してバッチ次元を追加
    tensor = transform(image).unsqueeze(0)
    
    return tensor
