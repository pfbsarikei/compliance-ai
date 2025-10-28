from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "✅ AI Compliance Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Create a DataFrame using the same column names as in training
    new_data = pd.DataFrame([{
        'Total Medication': data['Total Medication'],
        'NCD Medication': data['NCD Medication'],
        'Education': data['Education'],
        'Change in regimen': data['Change in regimen'],
        'Year on treatment': data['Year on treatment']
    }])

    prediction = model.predict(new_data)[0]
    confidence = model.predict_proba(new_data)[0][1]

    return jsonify({
        'Predicted Adherence': 'Yes' if prediction == 1 else 'No',
        'Confidence': round(float(confidence), 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

