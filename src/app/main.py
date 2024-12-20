import streamlit as st
import pandas as pd
from google.cloud import storage
import joblib

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

# Streamlit app
st.title("Gel Hardness Prediction App")

# Model selection
model_files = list_models()
selected_model = st.selectbox("Select a trained model", model_files)

# File upload
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Load the selected model
    model = load_model(selected_model)

    # Make predictions
    try:
        predictions = model.predict(df)
        st.success("Predictions generated successfully!")
        st.dataframe(pd.DataFrame(predictions, columns=["Predictions"]), use_container_width=True)

        # Save predictions (optional)
        save_to_gcs = st.checkbox("Save predictions to GCS?")
        if save_to_gcs:
            output_filename = st.text_input("Enter output filename (e.g., predictions.csv):", "predictions.csv")
            save = st.button("Save")
            if save:
                save_predictions(predictions, output_filename)
                st.success(f"Predictions saved to gs://{OUTPUT_BUCKET_NAME}/{output_filename}")

    except Exception as e:
        st.error(f"Error generating predictions: {e}")
