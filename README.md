# HoaXGY - Fake News Detection for Android (Machine Learning Module)

This repository contains the Machine Learning side of a Fake News Detection project.  
My role in the team is building and documenting the end-to-end ML pipeline, then exporting a lightweight model artifact for Android integration.

## Portfolio Summary

This project demonstrates:
- Text preprocessing for Indonesian news/hoax style content
- Baseline classical ML (TF-IDF + Logistic Regression)
- Deep learning text classifier (Keras)
- Export to TensorFlow Lite (`.tflite`) for mobile consumption
- OOP code organization for maintainability and handoff

## Project Structure

```text
fake-news-detection/
├── dataset/
│   ├── sample_dataset.xlsx
│   └── download_link.txt
├── demo/
│   └── app_demo.gif
├── models/
│   ├── baseline.pkl
│   ├── HoaXGY_model.keras
│   └── HoaXGY_model.tflite
├── notebooks/
│   └── HoaXGY.ipynb
├── src/
│   ├── __init__.py
│   ├── data_cleaner.py
│   ├── pipeline.py
│   └── models.py
├── main.py
├── requirements.txt
└── README.md
```

## Architecture (ML Flow)

1. Load dataset (`sample_dataset.xlsx` for demo mode)
2. Clean and normalize text fields
3. Split data into train/validation sets
4. Train baseline model for quick benchmark
5. Train deep learning model for final mobile export
6. Export Keras + TFLite models to `models/`
7. Evaluate and print final metrics (accuracy, ROC-AUC, report)

## Core Modules

- `src/data_cleaner.py`  
	OOP helper class for regex-based text preprocessing and source-specific cleanup.

- `src/pipeline.py`  
	Dataset loading, feature preparation, and train/validation splitting utilities.

- `src/models.py`  
	Baseline model and deep learning model classes, including training, evaluation, and TFLite export.

- `main.py`  
	Main orchestrator that runs the full pipeline and generates model artifacts for handoff.

## Setup and Run

### 1. Create virtual environment (recommended)

```bash
python -m venv .venv
```

### 2. Activate virtual environment

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training and export

```bash
python main.py
```

## Expected Outputs

After running `main.py`, the `models/` folder will contain generated artifacts such as:
- `baseline.pkl`
- `HoaXGY_model.keras`
- `HoaXGY_model.tflite`

The console output also reports:
- TFLite model size (MB)
- Validation accuracy
- ROC-AUC
- Classification report

## Dataset Notes

- `dataset/sample_dataset.xlsx` is a small, version-controlled sample used for demonstration.
- Original full datasets are intentionally excluded from this repository because of size.
- The source link/location for original datasets is documented in `dataset/download_link.txt`.

## Mobile Team Handoff

For Android integration, the key artifact is:
- `models/HoaXGY_model.tflite`

Suggested handoff bundle:
1. `HoaXGY_model.tflite`
2. Label definition (`0 = Real`, `1 = Fake`) used by the ML pipeline
3. Input preprocessing rule summary (lowercase + regex cleaning + tokenizer sequence/padding)

## Reproducibility

- Random seeds are fixed in the pipeline where relevant.
- The notebook version of experiments is available at `notebooks/HoaXGY.ipynb`.
- Running with the sample dataset provides a lightweight reproducible demo for review.

## My Contribution

In this group project, I handled the Machine Learning part:
- Data preprocessing strategy
- Baseline and deep learning experiments
- Model export to mobile-friendly format (TFLite)
- Documentation of the ML pipeline and handoff notes

## App Demo

Here's a quick demonstration of the Android app consuming the model for fake news detection:

<img src="demo/app_demo.gif" alt="App Demo" width="400">