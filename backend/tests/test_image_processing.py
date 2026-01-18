import pytest
from PIL import Image
import io
import torch
from app.utils.image_processing import preprocess_image

def test_preprocess_image_resnet():
    """ResNet用の前処理が正しく動作することを確認"""
    # ダミー画像作成
    img = Image.new('RGB', (500, 500), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    tensor = preprocess_image(img_bytes.read(), model_type="resnet152")
    
    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(tensor, torch.Tensor)

def test_preprocess_image_vit():
    """ViT用の前処理が正しく動作することを確認"""
    img = Image.new('RGB', (500, 500), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    tensor = preprocess_image(img_bytes.read(), model_type="vit")
    
    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(tensor, torch.Tensor)

def test_preprocess_invalid_image():
    """無効な画像でエラーが発生することを確認"""
    with pytest.raises(Exception):
        preprocess_image(b"invalid image data", model_type="resnet152")
