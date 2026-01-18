from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.classifier import ModelClassifier
from app.utils.image_processing import preprocess_image
from app.schemas.prediction import PredictionResponse, PredictionItem
from typing import List
import logging

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = logging.getLogger(__name__)

# モデルのキャッシュ（起動時に一度だけロード）
_models = {}

def get_model(model_type: str) -> ModelClassifier:
    """モデルを取得（キャッシュ機能付き）"""
    if model_type not in _models:
        try:
            _models[model_type] = ModelClassifier(model_type=model_type)
            logger.info(f"Loaded {model_type} model")
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return _models[model_type]

@router.post("/resnet152", response_model=PredictionResponse)
async def predict_resnet152(file: UploadFile = File(...)):
    """
    ResNet152モデルで画像分類を実行
    
    - **file**: 分類する画像ファイル（JPEG, PNG）
    """
    return await _predict(file, "resnet152")

@router.post("/vit", response_model=PredictionResponse)
async def predict_vit(file: UploadFile = File(...)):
    """
    Vision Transformer (ViT-B/16)モデルで画像分類を実行
    
    - **file**: 分類する画像ファイル（JPEG, PNG）
    """
    return await _predict(file, "vit")

async def _predict(file: UploadFile, model_type: str) -> PredictionResponse:
    """共通の予測処理"""
    # ファイル形式のチェック
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        # 画像を読み込み
        image_bytes = await file.read()
        
        # 前処理
        image_tensor = preprocess_image(image_bytes, model_type)
        
        # モデルを取得
        model = get_model(model_type)
        
        # 推論
        predictions = model.predict(image_tensor, top_k=5)
        
        # レスポンスを構築
        prediction_items = [
            PredictionItem(class_name=pred["class_name"], confidence=pred["confidence"])
            for pred in predictions
        ]
        
        return PredictionResponse(
            model_type=model_type,
            predictions=prediction_items
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
