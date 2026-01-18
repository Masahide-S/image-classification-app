from pydantic import BaseModel, Field, ConfigDict
from typing import List

class PredictionItem(BaseModel):
    """単一の予測結果"""
    class_name: str = Field(..., description="予測されたクラス名")
    confidence: float = Field(..., ge=0.0, le=1.0, description="確信度（0.0-1.0）")

class PredictionResponse(BaseModel):
    """予測APIのレスポンス"""
    model_type: str = Field(..., description="使用したモデルタイプ")
    predictions: List[PredictionItem] = Field(..., description="予測結果のリスト")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "resnet152",
                "predictions": [
                    {"class_name": "cat", "confidence": 0.95},
                    {"class_name": "dog", "confidence": 0.03},
                    {"class_name": "bird", "confidence": 0.01}
                ]
            }
        }
    )
