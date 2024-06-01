# model_evaluation.py

"""
Model Evaluation Module for Bagging for Model Stability

This module contains functions for evaluating the performance of trained machine learning models.

Techniques Used:
- Evaluation Metrics
- Visualization

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib

class ModelEvaluation:
    def __init__(self, model_dir, model_type='regression'):
        """
        Initialize the ModelEvaluation class.
        
        :param model_dir: str, directory containing the trained models
        :param model_type: str, type of model ('regression' or 'classification')
        """
        self.model_dir = model_dir
        self.model_type = model_type
        self.model = self.load_model('bagging_model.pkl')

    def load_model(self, model_name):
        """
        Load a trained model from a file.
        
        :param model_name: str, name of the model file
        :return: loaded model
        """
        model_path = os.path.join(self.model_dir, model_name)
        model = joblib.load(model_path)
        return model

    def load_data(self, data_filepath, target_column):
        """
        Load data from a CSV file and split into features and target.
        
        :param data_filepath: str, path to the CSV file
        :param target_column: str, name of the target column
        :return: tuple, features and target data
        """
        data = pd.read_csv(data_filepath)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def evaluate_regression(self, X_test, y_test):
        """
        Evaluate regression model performance.
        
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Explained Variance': explained_variance_score(y_test, y_pred)
        }
        return metrics

    def evaluate_classification(self, X_test, y_test):
        """
        Evaluate classification model performance.
        
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted')
        }
        return metrics

    def plot_predictions(self, y_test, y_pred, output_dir):
        """
        Plot the true vs predicted values.
        
        :param y_test: array, true values
        :param y_pred: array, predicted values
        :param output_dir: str, directory to save the plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.savefig(os.path.join(output_dir, 'true_vs_predicted.png'))
        plt.show()

    def evaluate(self, data_filepath, target_column, output_dir):
        """
        Execute the full evaluation pipeline.
        
        :param data_filepath: str, path to the input data file
        :param target_column: str, name of the target column
        :param output_dir: str, directory to save the evaluation results
        """
        # Load data
        X_test, y_test = self.load_data(data_filepath, target_column)

        # Evaluate model
        if self.model_type == 'regression':
            metrics = self.evaluate_regression(X_test, y_test)
        elif self.model_type == 'classification':
            metrics = self.evaluate_classification(X_test, y_test)
        else:
            raise ValueError("Invalid model_type. Choose 'regression' or 'classification'.")

        # Print evaluation metrics
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        # Plot predictions
        y_pred = self.model.predict(X_test)
        self.plot_predictions(y_test, y_pred, output_dir)

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    model_dir = 'models/'
    target_column = 'target'  # Example target column
    output_dir = 'results/evaluation/'
    os.makedirs(output_dir, exist_ok=True)

    model_type = 'regression'  # or 'classification'
    evaluator = ModelEvaluation(model_dir, model_type)

    # Evaluate the model
    evaluator.evaluate(data_filepath, target_column, output_dir)
    print("Model evaluation completed and results saved.")
