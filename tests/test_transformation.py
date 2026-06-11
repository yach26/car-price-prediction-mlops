import pandas as pd
import numpy as np
from src.components.data_transformation import FeatureEngineeringTransformer

def test_feature_engineering_transformer():
    # Construct mock raw dataframe
    df = pd.DataFrame({
        "transmission": ["8-Speed Automatic", "Manual 6-Speed", "CVT with overdrive", "Unknown Trans"],
        "mileage": [10.0, np.nan, 20.0, 30.0],
        "cylinders": [6.0, 4.0, np.nan, 8.0],
        "engine": ["v6", "i4", np.nan, "v8"],
        "fuel": ["Gasoline", "Gasoline", "Gasoline", "Gasoline"],
        "trim": ["Laredo", "Laredo", "Laredo", "Laredo"],
        "body": ["SUV", "SUV", "SUV", "SUV"],
        "doors": [4.0, 4.0, 4.0, 4.0],
        "exterior_color": ["White", "White", "White", "White"],
        "interior_color": ["Black", "Black", "Black", "Black"],
        "drivetrain": ["Four-wheel Drive", "Four-wheel Drive", "Four-wheel Drive", "Four-wheel Drive"],
        "type": ["New", "New", "New", "New"],
        "make": ["Jeep", "Jeep", "Jeep", "Jeep"],
        "name": ["car1", "car2", "car3", "car4"],
        "model": ["mod1", "mod2", "mod3", "mod4"],
        "description": ["desc1", "desc2", "desc3", "desc4"]
    })
    
    transformer = FeatureEngineeringTransformer()
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    
    # Verify simplify_transmission mappings
    expected_trans = ["Automatic", "Manual", "CVT", "Other"]
    assert list(transformed_df["transmission"]) == expected_trans
    
    # Verify missing numerical values imputation
    assert not transformed_df["mileage"].isnull().any()
    assert not transformed_df["cylinders"].isnull().any()
    
    # Verify missing categorical values imputation
    assert not transformed_df["engine"].isnull().any()
    
    # Verify that name, model, trim, and description columns are dropped
    for col in ["name", "model", "trim", "description"]:
        assert col not in transformed_df.columns
