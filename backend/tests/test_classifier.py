import pytest
from app.models.classifier import ModelClassifier

def test_model_classifier_initialization():
    """モデルが正しく初期化されることを確認"""
    classifier = ModelClassifier(model_type="resnet152")
    assert classifier.model is not None
    assert classifier.model_type == "resnet152"
    assert len(classifier.classes) > 0

def test_invalid_model_type():
    """無効なモデルタイプでエラーが発生することを確認"""
    with pytest.raises(ValueError):
        ModelClassifier(model_type="invalid_model")

def test_model_classes_loaded():
    """クラス名が正しく読み込まれることを確認"""
    classifier = ModelClassifier(model_type="resnet152")
    assert isinstance(classifier.classes, list)
    assert all(isinstance(c, str) for c in classifier.classes)
