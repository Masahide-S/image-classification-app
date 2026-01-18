import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """ヘルスチェックエンドポイントが200を返すことを確認"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_root_endpoint():
    """ルートエンドポイントが正しいメッセージを返すことを確認"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
