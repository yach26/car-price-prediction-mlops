import os
import pandas as pd
from src.components.data_ingestion import DataIngestion, DataIngestionConfig

def test_data_ingestion_config():
    config = DataIngestionConfig()
    assert config.raw_data_path == os.path.join("artifacts", "raw.csv")
    assert config.train_data_path == os.path.join("artifacts", "train.csv")
    assert config.test_data_path == os.path.join("artifacts", "test.csv")

def test_data_ingestion_execution():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    assert not train_df.empty
    assert not test_df.empty
    
    # Check shape splits (should be roughly 80/20)
    total_len = len(train_df) + len(test_df)
    assert abs(len(train_df) / total_len - 0.8) < 0.05
