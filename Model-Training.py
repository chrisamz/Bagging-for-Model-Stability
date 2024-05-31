# model_training.py

"""
Model Training Module for Bagging for Model Stability

This module contains functions for training machine learning models using bagging methods to enhance stability and robustness.

Techniques Used:
- Bagging
- Bootstrap Sampling
- Ensemble Methods

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- joblib

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.externals import joblib
import os

class ModelTraining:
    def __init__(self, model_type='regression'):
        """
        Initialize the ModelTraining class.
        
        :param model_type: str, type of model ('regression' or 'classification')
        """
        self.model_type = model_type
        if model_type == 'regression':
            self.model = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, random_state=42)
        elif model_type == 'classification':
            self.model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
        else:
            raise ValueError("Invalid model_type. Choose 'regression' or 'classification'.")

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def split_data(self, data, target_column, test_size=0.2):
        """
        Split the data into training and testing sets.
        
        :param data: DataFrame, input data
        :param target_column: str, name of the target column
        :param test_size: float, proportion of the data to include in the test split
        :return: tuple, training and testing data
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Train the bagging model.
        
        :param X_train: DataFrame, training features
        :param y_train: Series, training target
        """
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test data.
        
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        if self.model_type == 'regression':
            metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'Explained Variance': self.model.score(X_test, y_test)
            }
        elif self.model_type == 'classification':
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred)
            }
        return metrics

    def save_model(self, model_dir, model_name='bagging_model.pkl'):
        """
        Save the trained model to a file.
        
        :param model_dir: str, directory to save the model
        :param model_name: str, name of the model file
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    model_dir = 'models/'
    target_column = 'target'  # Example target column

    model_type = 'regression'  # or 'classification'
    trainer = ModelTraining(model_type)

    # Load data
    data = trainer.load_data(data_filepath)

    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(data, target_column)

    # Train model
    trainer.train_model(X_train, y_train)
    print("Model training completed.")

    # Evaluate model
    metrics = trainer.evaluate_model(X_test, y_test)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # Save model
    trainer.save_model(model_dir)
