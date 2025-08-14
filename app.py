from flask import Flask, render_template, request
import joblib
from check import url_features

app = Flask(__name__)

# Load model
model_data = joblib.load("model.pkl")
model = model_data["model"]
columns = model_data["columns"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form["url"]
    feats = url_features(url)
    import pandas as pd
    X = pd.DataFrame([feats])
    for c in columns:
        if c not in X.columns:
            X[c] = 0
    X = X[columns]
    pred = model.predict(X)[0]
    label = "Phishing URL" if pred == 1 else "Safe URL"
    return render_template("result.html", url=url, label=label)

if __name__ == "__main__":
    app.run(debug=True)
