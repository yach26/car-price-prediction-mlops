import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method")

        try:
            df = pd.read_csv(
                "notebook/data/vehicles_dataset.csv"
            )

            logging.info("Dataset read successfully")

            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        
        from src.components.data_transformation import DataTransformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        
        from src.components.model_trainer import ModelTrainer
        model_trainer = ModelTrainer()
        best_model_name, best_model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"Pipeline executed successfully. Best model: {best_model_name} with score: {best_model_score}")
    except Exception as e:
        print(f"Error in running pipeline: {e}")