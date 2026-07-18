# 🌐 English-to-Arabic Machine Translator

A full-stack graduation project that combines **data analysis**, **neural machine translation**, and a **web application** for translating English text to Arabic.

## 🏗️ Architecture

| Service | Technology | Port |
|---------|-----------|------|
| Frontend | React 18 | 3000 |
| Backend API | Django + DRF | 8000 |
| ML Worker | PyTorch + Transformers (GPU) | 8001 |
| MongoDB | mongo:7.0 | 27017 |
| SQL Server | mssql:2022 | 1433 |

## 🚀 Quick Start

```bash
# Clone and start all services
docker-compose up --build

# Access the app
# Frontend:  http://localhost:3000
# Backend:   http://localhost:8000/api/
# ML Worker: http://localhost:8001
```

## 📊 Data Pipeline (8-Step Workflow)

1. **Load & Explore** — Dataset structure, shape, encoding
2. **Duplicates** — Detect & remove duplicate sentence pairs
3. **Missing Values** — Handle empty/null translations
4. **Outliers** — Filter abnormal sentence lengths/ratios
5. **Visualizations** — 7+ charts for EDA dashboard
6. **Imbalance** — Domain distribution check
7. **Training** — Fine-tune Helsinki-NLP/opus-mt-en-ar
8. **Evaluation** — BLEU, chrF, TER metrics + baseline comparison

## 🧠 Model

- **Base**: `Helsinki-NLP/opus-mt-en-ar` (Marian MT)
- **Training**: fp16 mixed precision, batch 4 × 8 gradient accumulation
- **Target**: BLEU ≥ 25 on test set

## 📂 Project Structure

```
├── docker-compose.yml
├── backend/          # Django + DRF API
│   ├── api/          # REST endpoints
│   └── ml/           # ML pipeline (Steps 1-8)
├── ml_worker/        # GPU training/inference container
├── frontend/         # React dashboard
├── data/             # Datasets (volume mounted)
├── models/           # Saved model checkpoints
└── charts/           # Generated EDA charts
```

## 📋 Prerequisites

- Docker Desktop with WSL2 backend
- NVIDIA GPU + Container Toolkit (for training)
- 6GB+ VRAM recommended
