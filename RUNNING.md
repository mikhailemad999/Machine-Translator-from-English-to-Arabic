# 🌐 English-to-Arabic Machine Translator: Run & Train Guide

This guide describes how to run and train the English-to-Arabic Machine Translator project on a local Windows machine without Docker (Local Mode) or with Docker.

---

## 🛠️ Part 1: Local Development Mode (Recommended)

Local Mode runs the entire application stack directly on Windows. It bypasses SQL Server and MongoDB dependencies by utilizing **SQLite** and a **JSON-file-based fallback DB** (`local_mongo_data/`).

### 1. Prerequisites
- **Python 3.11 or 3.12** installed and added to `PATH`
- **Node.js 18+** installed (for the React dashboard)
- **NVIDIA GPU (optional but recommended)**: If an NVIDIA GPU is available (like your RTX 2060), the setup script automatically installs PyTorch with GPU acceleration (CUDA 12.1), which makes model training and inference 20-50x faster.

### 2. Auto-run Script (`start_local.bat`)
We have optimized and fixed `start_local.bat` to handle environment setup dynamically. When you run `start_local.bat`, it performs the following:
1. Verifies/creates a default local `.env` configuration.
2. Checks for Python and Node.js.
3. Automatically sets up a local virtual environment (`venv`).
4. **Detects your GPU**: Installs GPU-enabled PyTorch if `nvidia-smi` is detected; otherwise installs CPU PyTorch.
5. Installs backend dependencies.
6. Runs Django SQLite migrations (`db.sqlite3`).
7. Checks if the base machine translation model is downloaded; if not, downloads it (~300MB).
8. Launches three separate windows for:
   - **Frontend (React)** on [http://localhost:3000](http://localhost:3000)
   - **Backend API (Django)** on [http://localhost:8000/api/](http://localhost:8000/api/)
   - **ML Worker (Flask)** on [http://localhost:8001/](http://localhost:8001/)

**To run the stack:**
Double-click `start_local.bat` in your project folder, or run it via command line:
```cmd
start_local.bat
```

To stop all services, return to the main batch window and press any key (or press Ctrl+C). It will clean up all background processes automatically.

---

## 📊 Part 2: Step-by-Step Training & Pipeline Execution

The system features an **8-Step Machine Learning Pipeline** for English-to-Arabic Translation.

### Step 1: Download the Training Dataset
Run the download script to fetch sentence pairs from HuggingFace (`opus-100` and `opus_books` datasets). By default, it retrieves 50,000 sentence pairs and filters out duplicate translations.
```cmd
venv\Scripts\python.exe download_dataset.py
```
This saves the corpus to `data/en_ar_dataset.csv`.

### Step 2: Download the Pre-Trained Model
Download the base translator model (`Helsinki-NLP/opus-mt-en-ar`) from HuggingFace to your local directory for fine-tuning.
```cmd
venv\Scripts\python.exe download_model.py
```
This saves the model files to `models/opus-mt-en-ar/`.

### Step 3: Train / Fine-tune the Model
Train the translation model locally. The script reads the configuration parameters from `.env` (batch size, learning rate, FP16 precision, etc.) and trains the model.
```cmd
venv\Scripts\python.env train_model.py
```

#### Training Features & Parameters:
- **GPU Acceleration**: Uses CUDA and Mixed Precision (`FP16=True`) to train quickly on a 6GB VRAM GPU.
- **Early Stopping**: Halts training automatically when validation loss stops improving (patience parameter in `.env`).
- **Learning Curves**: Saves validation charts and loss trends to `charts/learning_curves.png`.
- **Saved Checkpoint**: The best model is saved at `models/best_model` and will be loaded automatically by the backend for future translations.

---

## 🏗️ Part 3: Alternative Docker Mode

If you prefer to run the project containerized with full MS SQL Server and MongoDB services:

1. Ensure **Docker Desktop** is running.
2. Ensure you have the **NVIDIA Container Toolkit** installed if you want GPU support in Docker.
3. Run the docker startup script:
   ```cmd
   start.bat
   ```
4. Access the apps:
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - Django API: [http://localhost:8000/api/](http://localhost:8000/api/)
   - ML Worker: [http://localhost:8001/](http://localhost:8001/)

---

## 🔍 Part 4: Project Structure

- `backend/` - Django API and database handlers.
- `ml_worker/` - Flask API wrapper around the translation PyTorch model.
- `frontend/` - React application providing a translation and training monitoring dashboard.
- `data/` - Training data storage folder.
- `models/` - Saved model checkpoints.
- `charts/` - Matplotlib training plots.
