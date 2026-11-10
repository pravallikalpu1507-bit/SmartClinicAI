from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def train_models(X, y):
    """Train KNN, Decision Tree and Naive Bayes; return accuracy dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "KNN":           KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes":   GaussianNB(),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        results[name] = round(acc * 100, 2)
        trained[name] = model

    return results, trained
