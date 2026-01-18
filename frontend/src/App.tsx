import { useState, useCallback } from 'react';
import { predictImage } from './api';
import type { PredictionResponse, ModelType } from './types';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [modelType, setModelType] = useState<ModelType>('resnet152');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);

      // プレビュー画像を生成
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('画像を選択してください');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await predictImage(selectedFile, modelType);
      setPrediction(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : '予測に失敗しました。サーバーが起動しているか確認してください。'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>画像分類アプリ</h1>
        <p>ResNet152またはViTモデルで画像を分類します</p>
      </header>

      <main className="main">
        <div className="upload-section">
          <div className="model-selector">
            <label>
              <input
                type="radio"
                name="model"
                value="resnet152"
                checked={modelType === 'resnet152'}
                onChange={(e) => setModelType(e.target.value as ModelType)}
              />
              ResNet152
            </label>
            <label>
              <input
                type="radio"
                name="model"
                value="vit"
                checked={modelType === 'vit'}
                onChange={(e) => setModelType(e.target.value as ModelType)}
              />
              Vision Transformer (ViT)
            </label>
          </div>

          <div className="file-input-wrapper">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="file-input"
              id="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              {selectedFile ? selectedFile.name : '画像を選択'}
            </label>
          </div>

          {previewUrl && (
            <div className="preview">
              <img src={previewUrl} alt="プレビュー" className="preview-image" />
            </div>
          )}

          <button
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            className="predict-button"
          >
            {loading ? '予測中...' : '予測する'}
          </button>
        </div>

        {error && (
          <div className="error">
            <p>{error}</p>
          </div>
        )}

        {prediction && (
          <div className="results">
            <h2>予測結果 ({prediction.model_type})</h2>
            <div className="predictions-list">
              {prediction.predictions.map((pred, index) => (
                <div key={index} className="prediction-item">
                  <div className="rank">#{index + 1}</div>
                  <div className="class-name">{pred.class_name}</div>
                  <div className="confidence-bar-wrapper">
                    <div
                      className="confidence-bar"
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                  <div className="confidence">
                    {(pred.confidence * 100).toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by FastAPI + PyTorch + React</p>
      </footer>
    </div>
  );
}

export default App;
