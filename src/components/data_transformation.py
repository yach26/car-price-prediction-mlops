import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to apply cleaning and transformations 
    exactly matching the Jupyter notebook's preprocessing steps:
    1. Impute numerical columns: mileage (mean), cylinders (mode).
    2. Impute categorical columns with mode: engine, fuel, transmission, trim, body, doors, exterior_color, interior_color.
    3. Simplify transmission column using simplify_transmission logic.
    4. Drop columns name, model, trim, and description to match final feature set X3.
    """
    def __init__(self):
        self.mileage_mean = None
        self.cylinders_mode = None
        self.categorical_modes = {}

    def fit(self, X, y=None):
        try:
            X_df = pd.DataFrame(X).copy()
            
            # Save imputation values calculated on train set to prevent leakage
            if 'mileage' in X_df.columns:
                self.mileage_mean = X_df['mileage'].mean()
            if 'cylinders' in X_df.columns:
                mode_val = X_df['cylinders'].mode()
                self.cylinders_mode = mode_val[0] if not mode_val.empty else 6.0
                
            categorical_cols = [
                'engine', 'fuel', 'transmission', 'trim', 'body', 
                'doors', 'exterior_color', 'interior_color'
            ]
            for col in categorical_cols:
                if col in X_df.columns:
                    mode_val = X_df[col].mode()
                    self.categorical_modes[col] = mode_val[0] if not mode_val.empty else "Other"
                    
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        try:
            X_df = pd.DataFrame(X).copy()
            
            # 1. Fill missing values
            if 'mileage' in X_df.columns and self.mileage_mean is not None:
                X_df['mileage'] = X_df['mileage'].fillna(self.mileage_mean)
            if 'cylinders' in X_df.columns and self.cylinders_mode is not None:
                X_df['cylinders'] = X_df['cylinders'].fillna(self.cylinders_mode)
                
            categorical_cols = [
                'engine', 'fuel', 'transmission', 'trim', 'body', 
                'doors', 'exterior_color', 'interior_color'
            ]
            for col in categorical_cols:
                if col in X_df.columns and col in self.categorical_modes:
                    X_df[col] = X_df[col].fillna(self.categorical_modes[col])
                    
            # 2. Simplify transmission
            def simplify_transmission(x):
                x = str(x).lower()
                if 'automatic' in x or 'a/t' in x:
                    return 'Automatic'
                elif 'manual' in x:
                    return 'Manual'
                elif 'cvt' in x:
                    return 'CVT'
                else:
                    return 'Other'
                    
            if 'transmission' in X_df.columns:
                X_df['transmission'] = X_df['transmission'].apply(simplify_transmission)
                
            # 3. Drop columns that are not used in final model training (errors='ignore' covers if some don't exist)
            cols_to_drop = ['name', 'model', 'trim', 'description']
            X_df = X_df.drop(columns=[col for col in cols_to_drop if col in X_df.columns], errors='ignore')
            
            return X_df
        except Exception as e:
            raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates the complete preprocessing pipeline containing the custom feature engineering transformer
        followed by scaled and encoded transformations using ColumnTransformer.
        """
        try:
            # Columns corresponding to df.drop(['name', 'model', 'trim'], axis=1)
            num_cols = ['year', 'cylinders', 'mileage', 'doors']
            ord_cols = ['engine', 'exterior_color', 'interior_color']
            oh_cols = ['fuel', 'transmission', 'body', 'drivetrain', 'type', 'make']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", Pipeline(steps=[
                        ("scaler", StandardScaler())
                    ]), num_cols),
                    
                    ("ord_pipeline", Pipeline(steps=[
                        ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                    ]), ord_cols),
                    
                    ("oh_pipeline", Pipeline(steps=[
                        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ]), oh_cols)
                ]
            )
            
            pipeline = Pipeline(steps=[
                ("feature_engineering", FeatureEngineeringTransformer()),
                ("preprocessor", preprocessor)
            ])
            
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Loaded train and test CSV files successfully.")
            logging.info("Applying cleaning steps: removing rows where target variable 'price' is null.")
            
            # Clean target variable nulls (matching notebook logic)
            train_df = train_df.dropna(subset=['price'])
            test_df = test_df.dropna(subset=['price'])
            
            target_column_name = "price"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], errors='ignore')
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], errors='ignore')
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Obtaining data preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()
            
            logging.info("Applying preprocessing pipeline on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine transformed features and target variable
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saving preprocessor object to disk.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
