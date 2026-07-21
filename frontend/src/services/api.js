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

/** List all uploaded datasets in the SQL database. */
export const listDatasets = () => api.get('/dataset/');

/** Get metadata and MongoDB count details for a specific dataset ID. */
export const getDataset = (id) => api.get(`/dataset/${id}/`);

/** Upload a dataset file (CSV/TSV/JSON) along with parsing configuration. */
export const uploadDataset = (formData) =>
  api.post('/dataset/upload/', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

/** Download a parallel sample dataset from HuggingFace opus-100 (ar-en). */
export const downloadSampleDataset = (maxSamples = 10000) =>
  api.post('/dataset/download-sample/', { max_samples: maxSamples });

// ---- Preprocessing APIs ----

/** Trigger the full preprocessing pipeline (deduplication, outliers, missing values) for a dataset. */
export const runPreprocessing = (datasetId) =>
  api.post('/preprocess/run/', { dataset_id: datasetId });

/** Get execution summary and results of a specific preprocessing run ID. */
export const getPreprocessingRun = (id) => api.get(`/preprocess/${id}/`);

/** List all preprocessing run histories for a particular dataset ID. */
export const listPreprocessingRuns = (datasetId) =>
  api.get(`/preprocess/list/${datasetId}/`);

// ---- EDA APIs ----

/** Retrieve full EDA report (shapes, distributions, encoding notes) for a dataset. */
export const getEdaReport = (datasetId) => api.get(`/eda/report/${datasetId}/`);

/** Get the absolute image URL of a generated EDA chart with optional cache-busting timestamp. */
export const getChartUrl = (filename, timestamp = '') =>
  `${API_BASE_URL}/eda/chart/${filename}${timestamp ? '?t=' + timestamp : ''}`;

/** List all generated chart filenames and URLs. */
export const listCharts = () => api.get('/eda/charts/');

// ---- Training APIs ----

/** Queue and start fine-tuning a seq2seq translation model using config hyperparameters. */
export const startTraining = (config) => api.post('/train/start/', config);

/** Check current status, active progress, and logs of a training job. */
export const getTrainingStatus = (jobId) => api.get(`/train/status/${jobId}/`);

/** List all training run jobs for a specific dataset ID. */
export const listTrainingJobs = (datasetId) =>
  api.get('/train/list/', { params: { dataset_id: datasetId } });

// ---- Evaluation APIs ----

/** Run sacrebleu/Qualitative evaluation metrics on a completed fine-tuned checkpoint. */
export const runEvaluation = (jobId) => api.post(`/evaluate/run/${jobId}/`);

/** Fetch existing evaluation report metrics and qualitative examples. */
export const getEvaluation = (jobId) => api.get(`/evaluate/${jobId}/`);

// ---- Translation APIs ----

/** Translate a single sentence of English text to Arabic. */
export const translateText = (text, modelPath = '') =>
  api.post('/translate/', { text, model_path: modelPath });

/** Batch translate a list of English sentences to Arabic. */
export const translateBatch = (texts, modelPath = '') =>
  api.post('/translate/batch/', { texts, model_path: modelPath });

/** Get recent translation requests logged in SQL. */
export const getTranslationHistory = (limit = 20) =>
  api.get('/translate/history/', { params: { limit } });

export default api;
