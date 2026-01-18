import axios from 'axios';
import { PredictionResponse, ModelType } from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const predictImage = async (
  file: File,
  modelType: ModelType
): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post<PredictionResponse>(
    `${API_BASE_URL}/predict/${modelType}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};
