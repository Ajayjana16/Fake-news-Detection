# Fake News Detection System using Machine Learning

A clean, academic research and demonstration platform for detecting fake and authentic news articles using supervised machine learning and natural language processing (NLP).

The core machine learning pipeline, vectorizer, and models were developed, trained, evaluated, and exported from a Google Colab notebook environment into standalone `.joblib` model artifacts. The web application directly consumes these saved artifacts without retraining or modifying the underlying model weights.

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Features](#2-features)
- [3. Technologies Used](#3-technologies-used)
- [4. Machine Learning Models](#4-machine-learning-models)
- [5. NLP Preprocessing Pipeline](#5-nlp-preprocessing-pipeline)
- [6. Label Mapping & Integrity](#6-label-mapping--integrity)
- [7. Project Structure](#7-project-structure)
- [8. Installation](#8-installation)
- [9. How to Run the Backend](#9-how-to-run-the-backend)
- [10. How to Run the Frontend](#10-how-to-run-the-frontend)
- [11. How Models Are Loaded](#11-how-models-are-loaded)
- [12. API Endpoints](#12-api-endpoints)
- [13. Model Evaluation Results](#13-model-evaluation-results)
- [14. Troubleshooting](#14-troubleshooting)

---

## 1. Project Overview

The objective of this project is to distinguish between genuine news reports and manufactured or misleading articles. Using the **ISOT Fake and Real News Dataset**, the training process in Google Colab subjected news texts to text cleaning, tokenization, lemmatization, and TF-IDF feature extraction before training three supervised classification algorithms.

This system provides:
1. A **Flask REST API** backend that applies identical NLP preprocessing and uses the serialized Colab `.joblib` models to infer news veracity.
2. A **Professional Academic ML Dashboard** displaying live classification, performance comparison tables, and visual charts loaded directly from `model_comparison_results.csv`.

---

## 2. Features

- **Google Colab Artifact Preservation**: Integrates the exact trained models (`logistic_regression_model.joblib`, `svm_model.joblib`, `random_forest_model.joblib`) and vectorizer (`tfidf_vectorizer.joblib`).
- **Identical 10-Step NLP Cleaning Pipeline**: Matches training tokenization, stopword removal, and WordNet lemmatization.
- **Model Switching**: Users can select which of the three trained models executes inference on any given input.
- **Dynamic Metrics Loading**: Reads `model_comparison_results.csv` on the fly to present actual accuracy, precision, recall, and F1 scores with zero fabricated statistics.
- **Health-Check Heartbeat**: Real-time status badge indicates backend availability (`Connected` vs `Offline`).
- **Interactive Sample Prompts**: One-click test samples (Reuters authentic news, clickbait fake news, and political news) for rapid verification.
- **Academic Dashboard**: Clean, responsive layout designed following university research presentation standards.

---

## 3. Technologies Used

- **Language**: Python 3.10+
- **Backend Framework**: Flask, Flask-CORS
- **Machine Learning & NLP**: scikit-learn, NLTK, Joblib, NumPy, Pandas
- **Frontend**: HTML5, CSS3 (IBM Plex typography), JavaScript (ES6+), Chart.js

---

## 4. Machine Learning Models

Three supervised classifiers were trained in the Google Colab experimental stage:

1. **Logistic Regression** (`logistic_regression_model.joblib`):
   - Fast, interpretable linear probability boundary.
   - Evaluates weighted linear combinations of n-gram TF-IDF scores.
2. **Support Vector Machine (SVM)** (`svm_model.joblib`):
   - Implemented via `LinearSVC` with maximum margin hyperplane separation.
   - Highly effective for high-dimensional, sparse text vectors.
3. **Random Forest** (`random_forest_model.joblib`):
   - Non-linear ensemble of decision trees.
   - Robust against overfitting with high feature dimensions.

*(Note: Decision Tree was evaluated during preliminary explorations in Colab but is excluded as a standalone production model in accordance with project syllabus requirements.)*

---

## 5. NLP Preprocessing Pipeline

To maintain feature consistency between training and inference, the backend reproduces the exact 10-step cleaning procedure:

1. **Type Normalization**: Cast input to string.
2. **Case Normalization**: Convert all characters to lowercase.
3. **URL Stripping**: Remove HTTP/HTTPS links and web URLs via regular expressions.
4. **HTML Removal**: Strip HTML tags and entity fragments.
5. **Non-Alphabetic Filtering**: Retain only alphabetic letters (`a-z`, `A-Z`) and single spaces.
6. **Whitespace Normalization**: Collapse multiple whitespace sequences into a single space.
7. **Tokenization**: Split text on whitespace into word tokens (`text.split()`).
8. **Stopword Elimination**: Filter out standard English stopwords using NLTK (`stopwords.words('english')`).
9. **Lemmatization**: Reduce tokens to root forms using NLTK's `WordNetLemmatizer`.
10. **Text Reconstruction**: Recombine processed tokens with single spaces.

When headline and article body are both provided, they are combined matching the training script:
```python
content = title + " " + text
```

Feature vectors are then generated using:
```python
tfidf_vectorizer.transform([cleaned_text])
```
*(Strictly `transform()`; `fit_transform()` is never called in production).*

---

## 6. Label Mapping & Integrity

Label mapping was verified directly from the trained model coefficients:
- **Class 0 &rarr; FAKE NEWS**: Model feature weights strongly associate negative values with markers such as `via`, `video`, `image via`, `breaking`, and sensationalist phrasing.
- **Class 1 &rarr; REAL NEWS**: Model feature weights strongly associate positive values with journalistic terms such as `reuters`, `said`, `washington (reuters)`, and calendar day reporting.

This mapping is defined in `backend/app.py`:
```python
LABEL_MAPPING = {
    0: "FAKE NEWS",
    1: "REAL NEWS"
}
```

---

## 7. Project Structure

```text
ml/
├── backend/
│   ├── app.py                            # Flask API server & static host
│   ├── preprocessing.py                  # 10-step NLP pipeline matching training
│   ├── models/                           # Exported Colab Joblib artifacts
│   │   ├── logistic_regression_model.joblib
│   │   ├── svm_model.joblib
│   │   ├── random_forest_model.joblib
│   │   └── tfidf_vectorizer.joblib
│   └── results/
│       └── model_comparison_results.csv  # Empirical evaluation metrics from Colab
│
├── frontend/
│   ├── index.html                        # Academic dashboard UI
│   ├── styles.css                        # University-style styling
│   └── app.js                            # Frontend controller, API fetch, & Chart.js
│
├── models/                               # Root model artifacts backup
├── results/                              # Root results CSV backup
├── requirements.txt                      # Production backend dependencies
└── README.md                             # Project documentation
```

---

## 8. Installation

1. Open PowerShell or a terminal inside the project root:
   ```bash
   cd "C:\Users\Ajay Jana\Desktop\ml"
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data resources (done automatically on first launch, or run manually):
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

---

## 9. How to Run the Backend

Run the Flask server from the project directory:

```bash
python backend/app.py
```

The server starts locally at:
```text
http://127.0.0.1:5000
```

Upon launch, the backend:
- Loads the TF-IDF vectorizer and all three ML models into memory once.
- Serves the `/api/health`, `/api/predict`, and `/api/metrics` REST endpoints.
- Directly hosts the academic dashboard frontend at `http://127.0.0.1:5000/`.

---

## 10. How to Run the Frontend

There are two flexible options:

### Option A: Direct Flask Hosting (Recommended)
Simply open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```
The Flask backend automatically serves the `frontend/index.html` dashboard with zero additional setup.

### Option B: Independent Web Server
If you prefer serving the frontend via a separate dev server (e.g., VS Code Live Server or Python HTTP server on port 3000/8080):
```bash
cd frontend
python -m http.server 3000
```
Open `http://localhost:3000`. The frontend includes automated CORS handling and automatically connects to the backend at `http://127.0.0.1:5000/api`.

---

## 11. How Models Are Loaded

At server initialization, `backend/app.py` resolves the models directory and loads each artifact using `joblib.load()`:

```python
import joblib

tfidf_vectorizer = joblib.load("backend/models/tfidf_vectorizer.joblib")
models["logistic_regression"] = joblib.load("backend/models/logistic_regression_model.joblib")
models["svm"] = joblib.load("backend/models/svm_model.joblib")
models["random_forest"] = joblib.load("backend/models/random_forest_model.joblib")
```

During request processing:
1. `raw_content` is preprocessed via `preprocess_text()`.
2. `tfidf_features = tfidf_vectorizer.transform([cleaned_content])` transforms the text into the 5,000-dimensional sparse feature space.
3. `selected_model.predict(tfidf_features)` executes classification.
4. The output integer (0 or 1) is mapped via `LABEL_MAPPING`.

---

## 12. API Endpoints

### 1. Health Check
- **Route**: `GET /api/health`
- **Response** (`200 OK`):
  ```json
  {
    "status": "ok",
    "models_loaded": ["logistic_regression", "svm", "random_forest"],
    "label_mapping": {
      "0": "FAKE NEWS",
      "1": "REAL NEWS"
    }
  }
  ```

### 2. News Prediction
- **Route**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "title": "U.S. House passes defense policy bill with bipartisan support",
    "text": "WASHINGTON (Reuters) - The U.S. House of Representatives on Tuesday passed...",
    "model": "logistic_regression"
  }
  ```
  *(Supported `model` values: `logistic_regression`, `svm`, `random_forest`)*
- **Response** (`200 OK`):
  ```json
  {
    "prediction": "REAL NEWS",
    "raw_class": 1,
    "model": "Logistic Regression",
    "model_key": "logistic_regression",
    "token_count": 18
  }
  ```

### 3. Model Comparison Metrics
- **Route**: `GET /api/metrics`
- **Response** (`200 OK`):
  ```json
  {
    "status": "ok",
    "source": "model_comparison_results.csv",
    "data": [
      {
        "model": "Logistic Regression",
        "accuracy": 0.9886948735169018,
        "precision": 0.9887371691409674,
        "recall": 0.9886948735169018,
        "f1_score": 0.9886971486670354
      },
      {
        "model": "SVM",
        "accuracy": 0.9959704499664204,
        "precision": 0.9959710738151386,
        "recall": 0.9959704499664204,
        "f1_score": 0.9959703114779428
      },
      {
        "model": "Random Forest",
        "accuracy": 0.9982090888739646,
        "precision": 0.9982108525260326,
        "recall": 0.9982090888739646,
        "f1_score": 0.9982091684090417
      }
    ]
  }
  ```

---

## 13. Model Evaluation Results

The values below are the exact empirical results recorded from the Google Colab evaluation run (`model_comparison_results.csv`):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 98.87% (0.9887) | 98.87% (0.9887) | 98.87% (0.9887) | 98.87% (0.9887) |
| **SVM (LinearSVC)** | 99.60% (0.9960) | 99.60% (0.9960) | 99.60% (0.9960) | 99.60% (0.9960) |
| **Random Forest** | 99.82% (0.9982) | 99.82% (0.9982) | 99.82% (0.9982) | 99.82% (0.9982) |

---

## 14. Troubleshooting

- **Missing Model Artifacts**: Ensure the `.joblib` files are situated in `backend/models/` (or `models/`).
- **NLTK Corpus Missing**: If `nltk_data` is not located in standard search paths, run:
  ```bash
  python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
  ```
- **Port Conflict**: If port 5000 is occupied by another service, change `port=5000` in the last line of `backend/app.py` to `port=5001`.
- **InconsistentVersionWarning**: A harmless warning may appear during unpickling due to minor differences between Colab's scikit-learn version and local environment. The warning is filtered automatically in `backend/app.py` without affecting prediction accuracy.
