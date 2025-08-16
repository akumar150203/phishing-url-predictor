from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url")
    result = model.predict([url])
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
