Final model weights: https://drive.google.com/drive/folders/1xAa8XrYL7A4Lr8k8HJ5-uwVuB6u4d-Vc?usp=sharing
# 🔬 Skin Lesion Segmentation & Classification

> Deep learning pipeline for automated skin lesion analysis · ISIC 2019 Dataset  
> **Data Science Practicum I · Regis University**

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-90.2%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%202019-lightblue)

---

## 01 | Project Overview

Skin cancer is one of the most common yet treatable cancers — if caught early. This project builds an end-to-end deep learning system that segments the lesion region from dermoscopy images and classifies it into one of 8 disease categories with over **90% accuracy**.

- 🔍 **Segment** the lesion region from raw dermoscopy images
- 🏷️ **Classify** lesions into 8 diagnostic categories (malignant & benign)
- 🌐 **Deploy** the model via a lightweight web application

---

## 02 | Disease Categories

The model classifies dermoscopy images into 8 categories:

| Code | Full Name | Type |
|------|-----------|------|
| **MEL** | Melanoma | 🔴 Malignant |
| **BCC** | Basal Cell Carcinoma | 🔴 Malignant |
| **SCC** | Squamous Cell Carcinoma | 🟠 Malignant |
| **AK** | Actinic Keratosis | 🟠 Pre-cancerous |
| **BKL** | Benign Keratosis-like Lesion | 🟢 Benign |
| **NV** | Melanocytic Nevus (Mole) | 🟢 Benign |
| **DF** | Dermatofibroma | 🟢 Benign |
| **VASC** | Vascular Lesion | 🟢 Benign |

---

## 03 | Repository Structure

```
Data-Science-Practicum-1/
│
├── app/               # Web application for model deployment (Flask / Streamlit)
├── data/              # Dataset handling & preprocessing scripts
├── notebooks/         # Jupyter notebooks — EDA, training, evaluation
├── results/           # Saved models, metrics, confusion matrices, plots
├── config.json        # Hyperparameters & training configuration
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## 04 | Model Architecture

- **Backbone:** ResNet50 pre-trained on ImageNet (Transfer Learning)
- **Input size:** 224 × 224 × 3 (RGB dermoscopy image)
- **Output:** Softmax probabilities over 8 disease classes
- **Framework:** TensorFlow / Keras

```
Input Image (224×224×3)
        ↓
ResNet50 Backbone (pre-trained weights, partially frozen)
        ↓
Global Average Pooling
        ↓
Dense 512 → ReLU → Dropout(0.5) → BatchNorm
        ↓
Dense 256 → ReLU → Dropout(0.3)
        ↓
Dense 8 → Softmax
        ↓
Prediction + Confidence Score
```

---

## 05 | Key Techniques & Why

| Challenge | Solution |
|-----------|----------|
| **Class imbalance** (NV=51%, DF=0.9%) | Weighted cross-entropy loss — rare class errors cost up to 21× more |
| **Limited medical data** | Data augmentation: rotation, flip, zoom, brightness shift |
| **Overfitting** | Dropout, L2 regularization, early stopping |
| **Training instability** | BatchNormalization + learning rate scheduling |
| **Insufficient training data** | Transfer learning from ImageNet (ResNet50 backbone) |

---

## 06 | Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **90.2%** |
| Melanoma Sensitivity | **90.9%** |
| Specificity | **98.6%** |
| AUC-ROC | **0.94** |
| F1-Score (macro avg) | **88.3%** |

> ⚠️ Sensitivity for melanoma is prioritized over raw accuracy. A false negative (missed cancer) is catastrophic. A false positive (unnecessary referral) is merely inconvenient.

---

## 07 | Getting Started

### Step 1: Clone the repo
```bash
git clone https://github.com/arjun7973/Data-Science-Practicum-1.git
cd Data-Science-Practicum-1
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download ISIC 2019 dataset
```
https://challenge.isic-archive.com/data/
```
Place images inside the `data/` directory.

### Step 4: Download model weights

📂 [Download Model Weights (Google Drive)](https://drive.google.com/drive/folders/1xAa8XrYL7A4Lr8k8HJ5-uwVuB6u4d-Vc?usp=sharing)

Place weights inside the `results/` directory.

### Step 5: Run notebooks
```bash
jupyter notebook notebooks/
```

### Step 6: Run the web app
```bash
cd app/
python app.py
```

---

## 08 | Configuration (`config.json`)

```json
{
  "model": "resnet50",
  "input_size": 224,
  "batch_size": 32,
  "epochs": 100,
  "learning_rate": 0.0001,
  "dropout_rate": 0.5,
  "optimizer": "adam",
  "loss": "weighted_categorical_crossentropy",
  "train_split": 0.70,
  "val_split": 0.15,
  "test_split": 0.15
}
```

---

## 09 | Limitations & Ethical Considerations

- ⚠️ **Not clinically validated** — academic / research project only
- 🎨 **Skin tone bias** — ISIC dataset underrepresents darker skin tones
- 📷 **Domain shift** — performance may vary across cameras and clinic settings
- 🏛️ **Regulatory** — FDA clearance required before any real-world medical use
- 👨‍⚕️ **Human in the loop** — model is a screening aid; doctors make all final decisions

---

## 10 | Future Work

- [ ] Grad-CAM heatmaps for model explainability and visual interpretability
- [ ] Multimodal model incorporating patient metadata (age, Fitzpatrick skin type)
- [ ] Fairness audit across demographic and skin tone groups
- [ ] Ensemble methods for improved robustness and confidence calibration
- [ ] Mobile deployment via TensorFlow Lite

---

## 11 | References

- [ISIC 2019 Challenge Dataset](https://challenge.isic-archive.com/)
- He et al. - *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016
- Esteva et al. - *Dermatologist-level classification of skin cancer with deep neural networks*, Nature 2017
- Codella et al. - *Skin Lesion Analysis Toward Melanoma Detection: ISIC 2019 Challenge*

---

## 👤 Author

**Arjun** · [@arjun7973](https://github.com/arjun7973) · Data Science Practicum I, Regis University

---

> *"This model doesn't replace doctors.*  
> *It gives everyone on earth access to a first line of defense*  
> *that currently only exists for the privileged few."*
