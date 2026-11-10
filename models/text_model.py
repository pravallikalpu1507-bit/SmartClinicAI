from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

def train_text_model(texts, labels):
    """Vectorize symptom text and train a Multinomial Naive Bayes classifier."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)
    return model, vectorizer

def predict_symptom(model, vectorizer, user_text):
    """Return predicted condition for a given symptom description."""
    text_vector = vectorizer.transform([user_text])
    prediction  = model.predict(text_vector)
    return prediction[0]

def predict_with_solution(model, vectorizer, user_text, df):
    """Return predicted condition AND its solution + severity from the dataset."""
    condition = predict_symptom(model, vectorizer, user_text)

    row = df[df["condition"] == condition]
    if not row.empty:
        solution = row.iloc[0]["solution"]
        severity = row.iloc[0]["severity"]
    else:
        solution = "Please consult a doctor for proper diagnosis."
        severity = "Unknown"

    return condition, severity, solution
