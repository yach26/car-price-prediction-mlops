import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
# Explicitly import the custom transformer so that pickle/dill can unserialize the preprocessor object successfully.
from src.components.data_transformation import FeatureEngineeringTransformer

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Loads the preprocessor and model pickle files and performs inference on the input features.
        """
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            # Load pickled objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transform features
            data_scaled = preprocessor.transform(features)
            
            # Run prediction
            preds = model.predict(data_scaled)
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Utility class to map web form input fields to a pandas DataFrame for prediction.
    """
    def __init__(self,
                 make: str,
                 model: str,
                 type: str,
                 year: int,
                 engine: str,
                 cylinders: float,
                 fuel: str,
                 mileage: float,
                 transmission: str,
                 trim: str,
                 body: str,
                 doors: float,
                 exterior_color: str,
                 interior_color: str,
                 drivetrain: str):
        self.make = make
        self.model = model
        self.type = type
        self.year = year
        self.engine = engine
        self.cylinders = cylinders
        self.fuel = fuel
        self.mileage = mileage
        self.transmission = transmission
        self.trim = trim
        self.body = body
        self.doors = doors
        self.exterior_color = exterior_color
        self.interior_color = interior_color
        self.drivetrain = drivetrain

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "make": [self.make],
                "model": [self.model],
                "type": [self.type],
                "year": [self.year],
                "engine": [self.engine],
                "cylinders": [self.cylinders],
                "fuel": [self.fuel],
                "mileage": [self.mileage],
                "transmission": [self.transmission],
                "trim": [self.trim],
                "body": [self.body],
                "doors": [self.doors],
                "exterior_color": [self.exterior_color],
                "interior_color": [self.interior_color],
                "drivetrain": [self.drivetrain]
            }
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)
