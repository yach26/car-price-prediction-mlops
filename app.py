from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Map form input values to CustomData object
            data = CustomData(
                make=request.form.get('make'),
                model=request.form.get('model'),
                type=request.form.get('type'),
                year=int(request.form.get('year')),
                engine=request.form.get('engine'),
                cylinders=float(request.form.get('cylinders')),
                fuel=request.form.get('fuel'),
                mileage=float(request.form.get('mileage')),
                transmission=request.form.get('transmission'),
                trim=request.form.get('trim'),
                body=request.form.get('body'),
                doors=float(request.form.get('doors')),
                exterior_color=request.form.get('exterior_color'),
                interior_color=request.form.get('interior_color'),
                drivetrain=request.form.get('drivetrain')
            )
            
            # Convert inputs into DataFrame
            pred_df = data.get_data_as_data_frame()
            
            # Predict price
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Format predicted price to ₹XX,XXX format
            formatted_price = f"{int(round(results[0])):,}"
            
            return render_template('home.html', result=formatted_price)
            
        except Exception as e:
            # Render error to page if something goes wrong
            return render_template('home.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
