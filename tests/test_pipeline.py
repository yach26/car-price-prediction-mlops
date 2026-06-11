import pandas as pd
import numpy as np
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

def test_custom_data_to_dataframe():
    # Verify CustomData formats form values to correct DataFrame columns
    data = CustomData(
        make="Jeep",
        model="Wagoneer",
        type="New",
        year=2024,
        engine="24V GDI DOHC Twin Turbo",
        cylinders=6.0,
        fuel="Gasoline",
        mileage=10.0,
        transmission="Automatic",
        trim="Series II",
        body="SUV",
        doors=4.0,
        exterior_color="White",
        interior_color="Global Black",
        drivetrain="Four-wheel Drive"
    )
    df = data.get_data_as_data_frame()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 15)
    assert df["make"].iloc[0] == "Jeep"
    assert df["mileage"].iloc[0] == 10.0

def test_predict_pipeline():
    # Verify PredictPipeline runs successfully on custom data DataFrame
    pipeline = PredictPipeline()
    data = CustomData(
        make="Jeep",
        model="Wagoneer",
        type="New",
        year=2024,
        engine="24V GDI DOHC Twin Turbo",
        cylinders=6.0,
        fuel="Gasoline",
        mileage=10.0,
        transmission="Automatic",
        trim="Series II",
        body="SUV",
        doors=4.0,
        exterior_color="White",
        interior_color="Global Black",
        drivetrain="Four-wheel Drive"
    )
    df = data.get_data_as_data_frame()
    predictions = pipeline.predict(df)
    
    assert len(predictions) == 1
    assert isinstance(predictions[0], (float, np.float32, np.float64, int))
    assert predictions[0] > 0
