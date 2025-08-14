import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import re, math

def shannon_entropy(s):
    from collections import Counter
    counts = Counter(s)
    return -sum(p/len(s) * math.log2(p/len(s)) for p in counts.values())

def url_features(u):
    parsed = urlparse(u)
    feats = {}
    feats["len_url"] = len(u)
    feats["num_dots"] = u.count(".")
    feats["entropy_url"] = shannon_entropy(u)
    feats["has_at"] = int("@" in u)
    feats["https"] = int(parsed.scheme == "https")
    return feats

if __name__ == "__main__":
    df = pd.read_csv("phishing_url.csv")
    df["label_bin"] = df["label"].map(lambda x: 1 if x.lower().startswith("p") else 0)
    X = df["url"].apply(url_features).apply(pd.Series)
    y = df["label_bin"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    joblib.dump({"model": model, "columns": list(X.columns)}, "model.pkl")
    print("Model saved as model.pkl")
