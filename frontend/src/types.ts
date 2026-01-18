export interface PredictionItem {
  class_name: string;
  confidence: number;
}

export interface PredictionResponse {
  model_type: string;
  predictions: PredictionItem[];
}

export type ModelType = 'resnet152' | 'vit';
