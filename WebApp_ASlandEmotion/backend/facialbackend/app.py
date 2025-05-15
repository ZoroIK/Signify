# Import necessary libraries
import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) to allow requests from the frontend
# In production, restrict 'allow_origins' to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development use only)
    allow_credentials=True,
    allow_methods=["POST"],  # Restrict to POST requests
    allow_headers=["*"],  # Allow all headers
)

# Define file paths for the saved model, label encoder, and scaler
MODEL_PATH = "mlp_emotion_model_balanced.pkl"
ENCODER_PATH = "label_encoder2.pkl"
SCALER_PATH = "scaler.pkl"

# Check if the model, encoder, and scaler files exist before proceeding
if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(SCALER_PATH)):
    print("Error: Model, encoder, or scaler file not found. Please ensure all files are present.")
    exit(1)  # Terminate if any required file is missing

# Attempt to load the pre-trained model, encoder, and scaler from disk
try:
    model = joblib.load(MODEL_PATH)  # Load the trained MLP model
    label_encoder = joblib.load(ENCODER_PATH)  # Load the label encoder for class names
    scaler = joblib.load(SCALER_PATH)  # Load the scaler used during training
    print("MLP model, label encoder, and scaler loaded successfully.")
except Exception as e:
    print(f"Failed to load model components: {e}")
    exit(1)  # Terminate on loading failure

# Define the expected input schema using Pydantic BaseModel
class LandmarkData(BaseModel):
    landmarks: list  # A flat list of 936 float values representing 468 (x, y) pairs

# Create the prediction API endpoint
@app.post("/predict")
async def predict(data: LandmarkData):
    # Ensure the input contains exactly 936 values
    if len(data.landmarks) != 468 * 2:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 936 values (x, y for 468 points). Got {len(data.landmarks)}."
        )

    try:
        # Convert input list to a NumPy array and scale it using the preloaded scaler
        input_array = np.array([data.landmarks])
        scaled_input = scaler.transform(input_array)

        # Make a prediction using the preloaded model
        prediction_encoded = model.predict(scaled_input)[0]

        # Decode the predicted class back to the human-readable label
        predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]

        # Return the prediction result as a JSON response
        return {
            "prediction": predicted_label
        }

    except Exception as e:
        # Handle unexpected errors during prediction
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")
