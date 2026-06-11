import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data into features and target.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Candidate models configured based on the Jupyter notebook
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                ),
                "Random Forest": RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1
                ),
                "XGBRegressor": XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42
                )
            }
            
            # Evaluate all models
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models
            )
            
            # Print comparative report in terminal
            print("\n" + "="*75)
            print("MODEL PERFORMANCE COMPARISON")
            print("="*75)
            print(f"{'Model Name':<25} | {'Train R2':<10} | {'Test R2':<10} | {'MAE':<10} | {'RMSE':<10}")
            print("-" * 75)
            for model_name, metrics in model_report.items():
                print(f"{model_name:<25} | {metrics['train_r2']:<10.4f} | {metrics['test_r2']:<10.4f} | {metrics['test_mae']:<10.2f} | {metrics['test_rmse']:<10.2f}")
            print("="*75 + "\n")
            
            # Automatically retrieve the best model name and score
            best_model_name = max(model_report, key=lambda k: model_report[k]['test_r2'])
            best_model_score = model_report[best_model_name]['test_r2']
            best_model = models[best_model_name]
            
            logging.info(f"Best model selected: {best_model_name} with Test R2: {best_model_score:.4f}")
            
            # Set minimum performance threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable R2 score (> 0.6)", sys)
                
            logging.info(f"Saving best model to: {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            return best_model_name, best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)
