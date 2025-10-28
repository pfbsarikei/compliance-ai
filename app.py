from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Compliance AI Predictor is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return jsonify({
        'adherence': int(prediction),
        'confidence': round(float(prob), 3)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
