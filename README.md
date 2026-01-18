# Image Classification App

ResNet152とVision Transformer (ViT)による画像分類Webアプリケーション

## 概要

このアプリケーションは、PyTorchを使用した深層学習モデル（ResNet152とViT）による画像分類を行うWebアプリケーションです。

### 技術スタック

**バックエンド:**
- FastAPI
- PyTorch
- torchvision
- timm (PyTorch Image Models)

**フロントエンド:**
- React
- TypeScript (予定)
- pnpm

## プロジェクト構造

```
image-classification-app/
├── backend/
│   ├── app/
│   │   ├── models/         # モデル定義
│   │   ├── routers/        # APIエンドポイント
│   │   ├── schemas/        # Pydanticスキーマ
│   │   ├── utils/          # ユーティリティ関数
│   │   └── main.py         # FastAPIアプリケーション
│   ├── models/             # 学習済みモデルファイル
│   ├── tests/              # テストコード
│   └── requirements.txt    # Python依存関係
└── frontend/               # Reactフロントエンド (開発中)

```

## セットアップ

### バックエンド

1. 依存関係のインストール:
```bash
cd backend
pip install -r requirements.txt
```

2. 学習済みモデルの配置:
`backend/models/`ディレクトリに以下のファイルを配置してください:
- `resnet152_best_model.pth`
- `vit_best_model.pth`
- `resnet152_classes.json`
- `vit_classes.json`

3. サーバーの起動:
```bash
uvicorn app.main:app --reload
```

### テスト

```bash
export PYTHONPATH=/path/to/image-classification-app/backend
pytest tests/ -v
```

## API エンドポイント

- `GET /` - API情報
- `GET /health` - ヘルスチェック
- `POST /predict/resnet152` - ResNet152で画像分類
- `POST /predict/vit` - ViTで画像分類
- `GET /docs` - Swagger UI

## 開発状況

- [x] バックエンドAPI実装
- [x] テスト作成・修正
- [x] GitHub Actions設定
- [ ] フロントエンド実装
- [ ] デプロイ設定

## ライセンス

MIT
