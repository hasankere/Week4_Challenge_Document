from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import logging
from typing import List
import joblib

# Load the pre-trained model (replace with your model path)
model_path = "C:\\Users\\Hasan\\Desktop\\EDA\\Week4_Challenge_Document\\notebooks\\optimized_random_forest_model_quick.pkl"
model = tf.keras.models.load_model(model_path)

# Create the FastAPI app
app = FastAPI()

# Set up logging (optional but useful for debugging)
logging.basicConfig(level=logging.INFO)

# Define input data model (using Pydantic to define the expected format)
class PredictionRequest(BaseModel):
    features: List[float]  # Input features, you can modify this to match your input structure

# Define the prediction endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        # Preprocess the input data (reshape it to match the model input shape)
        input_data = np.array(request.features).reshape(1, -1)  # Reshape as needed (example: [n_features, n_timesteps])
        
        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        
        # Return the prediction
        return {"prediction": prediction.tolist()}  # Convert numpy array to list for JSON response
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Define a health check endpoint
@app.get("/health/")
def health_check():
    return {"status": "OK"}
