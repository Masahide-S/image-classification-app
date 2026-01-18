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
- React 19
- TypeScript
- Vite
- pnpm
- Axios

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
└── frontend/               # Reactフロントエンド
    ├── src/
    │   ├── api.ts          # APIクライアント
    │   ├── types.ts        # TypeScript型定義
    │   ├── App.tsx         # メインコンポーネント
    │   └── App.css         # スタイル
    └── package.json

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

### フロントエンド

1. 依存関係のインストール:
```bash
cd frontend
pnpm install
```

2. 環境変数の設定:
```bash
cp .env.example .env
```

3. 開発サーバーの起動:
```bash
pnpm dev
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
- [x] フロントエンド実装
- [ ] デプロイ設定

## 使い方

1. バックエンドサーバーを起動:
```bash
cd backend
uvicorn app.main:app --reload
```

2. フロントエンドサーバーを起動（別ターミナル）:
```bash
cd frontend
pnpm dev
```

3. ブラウザで`http://localhost:5173`を開く

4. 画像をアップロードして、モデルを選択して予測を実行

## スクリーンショット

### メイン画面
- 画像アップロード機能
- モデル選択（ResNet152/ViT）
- リアルタイムプレビュー

### 予測結果
- Top-K予測結果の表示
- 信頼度の可視化（プログレスバー）
- レスポンシブデザイン

## ライセンス

MIT
