# model_deployment.py

"""
Model Deployment Module for Bagging for Model Stability

This module contains functions for deploying trained machine learning models using Flask for serving predictions.

Libraries/Tools:
- Flask
- pandas
- numpy
- scikit-learn
- joblib

"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os

app = Flask(__name__)

MODEL_DIR = 'models/'
model = None

def load_model(model_name):
    """
    Load a trained model from a file.
    
    :param model_name: str, name of the model file
    :return: loaded model
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    model = joblib.load(model_path)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make predictions using the trained model.
    Expects a JSON payload with input data.
    
    :return: JSON response with predictions
    """
    try:
        data = request.json
        data_df = pd.DataFrame(data)
        predictions = model.predict(data_df)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    model_name = 'bagging_model.pkl'
    model = load_model(model_name)
    app.run(host='0.0.0.0', port=5000)
