from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict_router
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Image Classification API",
    description="ResNet152とViTによる画像分類API",
    version="1.0.0"
)

# CORS設定（React用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Viteのデフォルトポート
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(predict_router)

@app.get("/")
async def root():
    return {
        "message": "Image Classification API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "resnet152": "/predict/resnet152",
            "vit": "/predict/vit"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
