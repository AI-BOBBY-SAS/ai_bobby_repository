import streamlit as st
import pandas as pd
from google.cloud import storage
import joblib
from loguru import logger

# Replace with your GCP project ID and bucket names
PROJECT_ID = "ai-bobby-poc"
MODEL_BUCKET_NAME = "ai-bobby-gel-hardness-models"  # Bucket for models
OUTPUT_BUCKET_NAME = "ai-bobby-gel-hardness-predictions"  # Bucket for predictions

# Initialize Google Cloud Storage client
storage_client = storage.Client(project=PROJECT_ID)
model_bucket = storage_client.bucket(MODEL_BUCKET_NAME)
output_bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)

# Function to list models in the GCS bucket
def list_models():
    blobs = model_bucket.list_blobs()
    model_files = [blob.name for blob in blobs if blob.name.endswith('.joblib')]
    return model_files

# Function to load a model from GCS
def load_model(model_file):
    blob = model_bucket.blob(model_file)
    blob.download_to_filename('model.joblib')
    model = joblib.load('model.joblib')
    return model

# Function to save predictions to GCS
def save_predictions(predictions, output_file):
    output_df = pd.DataFrame(predictions, columns=["Predictions"])
    output_blob = output_bucket.blob(output_file)
    output_blob.upload_from_string(output_df.to_csv(index=False), content_type="text/csv")

def reformat_predictions(predictions: list[int]) -> list[str]:
    predictions_map = {
        0: "Soft",
        1: "Medium",
        2: "Hard"
    }
    return [predictions_map[prediction] for prediction in predictions]


# Streamlit app
st.title("🤖 Gel Hardness Prediction App 🤖")
st.markdown(
    """
    ---

    ## How to use this application

    1. Select a trained model from the dropdown menu below. 
    The list of models available are the ones stored in the Google Cloud Storage bucket `ai-bobby-gel-hardness-models`.
    2. Upload a CSV file with the characteristics of the gel that were used to train the model.
    The CSV file should have the same columns as the training data (see `data/test/example_input.csv` in the github repo for an example).
    3. Click the "Predict" button to generate the predictions.
    ---

    """
)

# Model selection
model_files = list_models()
st.markdown("#### Select a trained model from the bucket:")
selected_model = st.selectbox("model selector", model_files, label_visibility="hidden")

# File upload
st.markdown("#### Upload a CSV file for prediction:")
uploaded_file = st.file_uploader("file selector", type="csv", label_visibility="hidden")

if uploaded_file is not None:
    # Read the CSV file
    logger.info("File uploaded.")
    df = pd.read_csv(uploaded_file)
    logger.info(f"Dataframe created. Columns: {list(df.columns)}")

    # Load the selected model
    model = load_model(selected_model)
    logger.info(f"Model {selected_model} loaded.")

    predictions_button = st.button("Predict")
    if predictions_button:
        # Make predictions
        try:
            logger.info("Predicting...")
            predictions = model.predict(df)
            reformatted_predictions = reformat_predictions(predictions)
            logger.info("Predictions generated successfully!")

            st.success("Predictions generated successfully!")
            logger.info("Displaying predictions...")
            st.dataframe(data=pd.DataFrame(reformatted_predictions, columns=["Predictions"]), use_container_width=True)
            logger.info("Predictions displayed successfully!")
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            logger.exception(f"Error generating predictions: {e}")
