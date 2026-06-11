import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a python object using dill serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Saved object to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a python object using dill serialization.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains multiple models and returns evaluation reports on both train and test data.
    """
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            report[model_name] = {
                "train_r2": train_r2,
                "test_r2": test_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse
            }
            logging.info(f"{model_name} - Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}, MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
            
        return report
    except Exception as e:
        raise CustomException(e, sys)
