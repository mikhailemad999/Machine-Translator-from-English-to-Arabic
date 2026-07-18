/**
 * API Service Layer — Axios instance for backend communication.
 */
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes (training can take long)
  headers: {
    'Content-Type': 'application/json',
  },
});

// ---- Dataset APIs ----
export const listDatasets = () => api.get('/dataset/');
export const getDataset = (id) => api.get(`/dataset/${id}/`);
export const uploadDataset = (formData) =>
  api.post('/dataset/upload/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
export const downloadSampleDataset = (maxSamples = 10000) =>
  api.post('/dataset/download-sample/', { max_samples: maxSamples });

// ---- Preprocessing APIs ----
export const runPreprocessing = (datasetId) =>
  api.post('/preprocess/run/', { dataset_id: datasetId });
export const getPreprocessingRun = (id) => api.get(`/preprocess/${id}/`);
export const listPreprocessingRuns = (datasetId) =>
  api.get(`/preprocess/list/${datasetId}/`);

// ---- EDA APIs ----
export const getEdaReport = (datasetId) => api.get(`/eda/report/${datasetId}/`);
export const getChartUrl = (filename) => `${API_BASE_URL}/eda/chart/${filename}`;
export const listCharts = () => api.get('/eda/charts/');

// ---- Training APIs ----
export const startTraining = (config) => api.post('/train/start/', config);
export const getTrainingStatus = (jobId) => api.get(`/train/status/${jobId}/`);
export const listTrainingJobs = (datasetId) =>
  api.get('/train/list/', { params: { dataset_id: datasetId } });

// ---- Evaluation APIs ----
export const runEvaluation = (jobId) => api.post(`/evaluate/run/${jobId}/`);
export const getEvaluation = (jobId) => api.get(`/evaluate/${jobId}/`);

// ---- Translation APIs ----
export const translateText = (text, modelPath = '') =>
  api.post('/translate/', { text, model_path: modelPath });
export const translateBatch = (texts, modelPath = '') =>
  api.post('/translate/batch/', { texts, model_path: modelPath });
export const getTranslationHistory = (limit = 20) =>
  api.get('/translate/history/', { params: { limit } });

export default api;
