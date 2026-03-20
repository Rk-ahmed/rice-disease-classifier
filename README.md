---
title: Rice Disease Classifier
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🌾 Rice Leaf Disease Classifier

> **MSc Thesis Project — East West University**
> *Deep Learning Architectures Showdown: Revealing the Ideal Model for Rice Disease Image Classification*
>
> **Author:** Md. Rakib Ahmed | **Supervisor:** Dr. Mohammad Rezaul Karim
> **Program:** MSc in Data Science and Analytics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Free Access](https://img.shields.io/badge/Access-Free%20for%20Everyone-brightgreen)]()

---

## 📖 Overview

This project implements a production-grade deep learning system for **automated classification of rice leaf diseases**. It compares four state-of-the-art CNN architectures using transfer learning, and deploys the best-performing model as a free, publicly accessible web application.

**Detectable Diseases:**
| Disease | Pathogen | Visual Symptoms |
|---------|----------|-----------------|
| 🔴 Bacterial Blight | *Xanthomonas oryzae* | Yellow, water-soaked lesions; wilting |
| 🟠 Blast | *Magnaporthe oryzae* | Diamond-shaped lesions; gray centers |
| 🟤 Brown Spot | *Bipolaris oryzae* | Small dark brown circular spots |
| 🟡 Tungro | RTBV + RTSV (viral) | Yellow-orange leaves; stunted growth |

**Dataset:** 5,932 expert-annotated RGB images (source: Sethy et al., 2020)

---

## 🏆 Key Results

| Model | Test Accuracy | F1-Score | AUC-ROC |
|-------|--------------|----------|---------|
| **VGG16** ⭐ | **100%** | **1.000** | **1.000** |
| Xception | 100% | 1.000 | 1.000 |
| InceptionV3 | 97.29% | 0.9699 | 0.999 |
| ResNet50 | 81.75% | 0.8086 | 0.960 |

> **VGG16 selected as the deployment model** due to its perfect classification performance and stable training behavior. Statistical significance confirmed via McNemar's Test (p < 0.05).

---

## 📁 Project Structure

```
rice-disease-classifier/
│
├── 📂 configs/
│   └── config.yaml              ← All project settings
│
├── 📂 src/
│   ├── 📂 data/
│   │   ├── preprocessor.py      ← Duplicate removal + splitting
│   │   └── data_loader.py       ← Data generators with augmentation
│   ├── 📂 models/
│   │   ├── build_model.py       ← VGG16, ResNet50, InceptionV3, Xception
│   │   └── train.py             ← Training script with K-Fold CV
│   ├── 📂 evaluation/
│   │   └── evaluate.py          ← Metrics + statistical tests
│   ├── 📂 visualization/
│   │   └── plots.py             ← Training curves, confusion matrix, ROC, Grad-CAM
│   └── 📂 utils/
│       ├── config_loader.py
│       ├── logger.py
│       └── seed.py
│
├── 📂 app/
│   ├── main.py                  ← FastAPI routes and server
│   ├── predictor.py             ← Model inference + Grad-CAM
│   ├── 📂 templates/
│   │   └── index.html           ← Web UI (drag and drop upload)
│   └── 📂 static/
│
├── 📂 notebooks/
│   └── rice_disease_training_colab.ipynb
│
├── 📂 tests/
│   └── test_model.py
│
├── 📂 outputs/
│   ├── models/                  ← Saved .keras model files
│   ├── plots/                   ← Training curves, confusion matrices
│   └── reports/                 ← Metrics and statistical test results
│
├── app.py                       ← Hugging Face entry point
├── Dockerfile                   ← Container configuration
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/rakib-ahmed/rice-disease-classifier.git
cd rice-disease-classifier
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Dataset
```
data/raw/Rice Leaf Disease Images/
    Bacterialblight/
    Blast/
    Brownspot/
    Tungro/
```

### 5. Run the Web Application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open your browser: **http://localhost:8000**

---

## 🧠 Model Architecture

All models follow the same transfer learning strategy:

```
Pre-trained Base (ImageNet weights, partially frozen)
         ↓
GlobalAveragePooling2D
         ↓
Dense(1024, relu)
         ↓
Dropout(0.5)
         ↓
Dense(4, softmax)  ← 4 disease classes
```

### Why VGG16 Was Selected

| Reason | Explanation |
|--------|-------------|
| Perfect accuracy | 100% on test set |
| Training stability | Consistent across all 5 CV folds |
| Simplicity | Easier to interpret and deploy |
| Statistical evidence | McNemar's test p < 0.05 vs all others |

---

## 📊 Statistical Validation

Four statistical methods used to rigorously compare models:

| Test | Purpose | Result |
|------|---------|--------|
| McNemar's Test | Compare error patterns on test set | VGG16 significantly better (p < 0.05) |
| Wilcoxon Signed-Rank | Compare CV fold scores | VGG16 folds significantly higher |
| Paired t-Test | Parametric fold comparison | Confirms Wilcoxon findings |
| Bootstrap 95% CI | Uncertainty bounds on accuracy | Tight CI confirms reliability |

---

## 📈 Data Augmentation

Applied only to training data:

| Technique | Value | Purpose |
|-----------|-------|---------|
| Rotation | 40 degrees | Handles varying leaf orientations |
| Horizontal flip | Random | Doubles effective dataset size |
| Width/height shift | 20% | Simulates partial leaf views |
| Zoom | 20% | Handles varying distances |
| Shear | 0.2 | Simulates perspective distortion |
| Brightness | 0.5 to 1.5 | Handles different lighting |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Server health check |
| POST | `/predict` | Predict disease from image |
| POST | `/predict-with-gradcam` | Predict + Grad-CAM heatmap |
| GET | `/classes` | List all disease classes |

### Example Request
```python
import requests

with open("rice_leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )

result = response.json()
print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.1%}")
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📚 References

- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.*
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition.*
- Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions.*
- Szegedy, C., et al. (2016). *Rethinking the Inception Architecture.*
- Sethy, P.K., et al. (2020). *Deep feature based rice leaf disease identification using support vector machine.*
- Selvaraju, R.R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

*"Dedicated to rice farmers who tirelessly sustain millions of lives."*