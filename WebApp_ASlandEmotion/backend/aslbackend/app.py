# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Define the list of possible output labels from the ASL model
# Includes 26 letters and two special tokens for space and delete
ASL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) to allow access from the frontend
# For production, restrict 'allow_origins' to trusted domains only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept requests from all domains (development only)
    allow_credentials=True,
    allow_methods=["POST"],  # Only allow POST requests
    allow_headers=["*"],  # Accept all headers
)

# Define the path to the trained TensorFlow model file
MODEL_PATH = "asl_cnn_2D_model.h5"

# Check if the model file exists before proceeding
if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found. Please make sure 'asl_cnn_2D_model.h5' is in the project directory.")
    exit(1)  # Exit if the model file is missing

# Attempt to load the TensorFlow model from the given path
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("ASL 2D CNN model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)  # Exit on model loading failure

# Define the structure of the expected input using Pydantic
class LandmarkData(BaseModel):
    landmarks: list  # A flat list of 42 float values (21 x, y coordinate pairs)

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: LandmarkData):
    # Validate input length: should be exactly 42 values (21 x, y pairs)
    if len(data.landmarks) != 42:
        raise HTTPException(
            status_code=400,
            detail="Expected 42 landmark values (x, y for 21 points)."
        )

    try:
        # Convert the list into a NumPy array and reshape for model input
        input_data = np.array([data.landmarks])

        # Run the model to get prediction probabilities
        predictions = model.predict(input_data, verbose=0)[0]

        # Identify the index of the class with the highest probability
        label_index = int(np.argmax(predictions))

        # Extract the confidence score for the predicted label
        confidence = float(np.max(predictions))

        # If confidence is too low, return "Unknown" as the prediction
        if confidence < 0.2:
            return {
                "prediction": "Unknown",
                "confidence": confidence
            }

        # Retrieve the corresponding label from ASL_LETTERS
        predicted_letter = ASL_LETTERS[label_index] if label_index < len(ASL_LETTERS) else "?"

        # Return the predicted label and confidence score
        return {
            "prediction": predicted_letter,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        # Handle any unexpected errors during prediction
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Error: {str(e)}"
        )
