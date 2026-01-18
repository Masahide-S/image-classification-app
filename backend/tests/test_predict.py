import pytest
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io
import json

client = TestClient(app)

@pytest.fixture
def sample_image():
    """テスト用のダミー画像を生成"""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def test_predict_resnet152(sample_image):
    """ResNet152で予測が正しく動作することを確認"""
    files = {"file": ("test.jpg", sample_image, "image/jpeg")}
    response = client.post("/predict/resnet152", files=files)

    # デバッグ用：レスポンスの内容を出力
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Body: {response.text}")

    if response.status_code != 200:
        print(f"Error Detail: {response.json()}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0  # 少なくとも1つの予測があること
    assert len(data["predictions"]) <= 5  # 最大5個
    assert "class_name" in data["predictions"][0]
    assert "confidence" in data["predictions"][0]

def test_predict_vit(sample_image):
    """ViTで予測が正しく動作することを確認"""
    files = {"file": ("test.jpg", sample_image, "image/jpeg")}
    response = client.post("/predict/vit", files=files)

    # デバッグ用：レスポンスの内容を出力
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Body: {response.text}")

    if response.status_code != 200:
        print(f"Error Detail: {response.json()}")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) > 0  # 少なくとも1つの予測があること
    assert len(data["predictions"]) <= 5  # 最大5個
    assert "class_name" in data["predictions"][0]
    assert "confidence" in data["predictions"][0]

def test_predict_no_file():
    """ファイルなしでエラーが返ることを確認"""
    response = client.post("/predict/resnet152")
    assert response.status_code == 422  # Validation Error

def test_predict_invalid_model():
    """無効なモデルタイプでエラーが返ることを確認"""
    sample_img = Image.new('RGB', (224, 224), color='blue')
    img_bytes = io.BytesIO()
    sample_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict/invalid", files=files)
    assert response.status_code == 404
