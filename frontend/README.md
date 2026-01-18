# Image Classification Frontend

ResNet152とViTによる画像分類のフロントエンドアプリケーション

## セットアップ

1. 依存関係のインストール:
```bash
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

## 機能

- 画像のアップロード
- モデル選択（ResNet152またはViT）
- リアルタイム予測
- 予測結果の可視化（Top-K）

## 技術スタック

- React 19
- TypeScript
- Vite
- Axios
- CSS3 (グラデーション、アニメーション)

## ビルド

```bash
pnpm build
```

## プレビュー

```bash
pnpm preview
```

## API接続

バックエンドサーバーが起動していることを確認してください：

```bash
cd ../backend
uvicorn app.main:app --reload
```

デフォルトでは`http://localhost:8000`に接続します。変更する場合は`.env`ファイルを編集してください。
