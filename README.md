# ⚕ SmartClinic AI — Disease Prediction System

An ML-powered web app that predicts a possible condition, its severity, and recommended actions from symptoms typed in plain English.

Project for CAP258 (Machine Learning Basics) and CAP263 (Python Programming), Lovely Professional University.

## Features
- Text model: CountVectorizer + Multinomial Naive Bayes on symptom descriptions
- Structured model: KNN, Decision Tree and Gaussian Naive Bayes on a heart disease dataset
- Accuracy comparison chart (Matplotlib)
- Streamlit web app with Symptom Checker, Model Performance and Dataset tabs

## Model Results (heart.csv)
| Model | Accuracy |
|---|---|
| KNN | 63.9% |
| Decision Tree | ~72% |
| Naive Bayes | 85.2% |

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
This is a university project and is not a substitute for professional medical advice.

Author: Pravallika Puppala