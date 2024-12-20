from kfp.v2 import dsl
from kfp.v2.dsl import Dataset, Input, Model, Output, Metrics, component


@component(
    packages_to_install=[
        "pandas==1.5.3",
        "numpy==1.24.4",
        "google-cloud-storage==2.19.0",
        "loguru==0.7.3"
    ],
    base_image="python:3.10"
)
def prepare_data(
    dataset_file_name: str,
    prepared_data: Output[Dataset],
    prepared_labels: Output[Dataset],
    project_id: str = "ai-bobby-poc",
):
    from google.cloud import storage
    import pandas as pd
    from loguru import logger

    def categorize_gel_strength(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Categorizes a continuous column into gel strength categories.

        Parameters:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to be transformed.

        Returns:
        pd.DataFrame: DataFrame with a new column `<column_name>_category` containing the categories.
        """
        # Define the bins and labels
        bins = [0, 1000, 5000, 1000000]
        labels = [0, 1, 2] # Soft, Firm, Rigid

        # Create a new column for the categories
        category_column_name = f"{column_name}_category"
        dataframe[category_column_name] = pd.cut(
            dataframe[column_name], bins=bins, labels=labels, include_lowest=True
        )

        return dataframe


    logger.info(f"Downloading dataset: {dataset_file_name}")
    # Download dataset from GCS path
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket('ai-bobby-gel-hardness-datasets')
    blob = bucket.blob(dataset_file_name)
    blob.download_to_filename(dataset_file_name)
    df = pd.read_csv(dataset_file_name)
    logger.info("Dataset downloaded.")

    logger.info("Changing column types and dropping unnecessary columns...")
    # Define categorical and numerical columns
    categorical_columns = ['Protein codes', 'Type of salt', 'Additives', 'Treatment code']
    numerical_columns = [
        'Samples stored (°C)',
        'ionic strength (M)', 
        'Additives Concentration (%)',
        'Protein Concentration (%)',
        'pH',
        'Heating temperature (°C) for gel preparation',
        'Heating/hold time (min)',
        'Hardness/firmness/strength (g)'
    ]

    # Convert datatypes
    for col in categorical_columns:
        df[col] = df[col].astype('object')
    for col in numerical_columns:
        df[col] = df[col].astype('float64')

    # Drop unnecessary columns
    columns_to_drop = [
        'Citation',
        'Citation Link',
        'Protein',
        'Treatment condition code',
        'Treatment condition value',
        'Treatment temperature ( °C)',
        'Treatment time (min)',
        'Storage time (h)',
        'If a gel can be formed (0-1)',
    ]
    df_clean = df.drop(columns=columns_to_drop, axis=1)
    logger.info("Column types changed and unnecessary columns dropped.")

    logger.info("Defining X and y...")
    # Process target variable
    df_clean = df_clean.dropna(subset=['Hardness/firmness/strength (g)'])
    df_clean = categorize_gel_strength(df_clean, 'Hardness/firmness/strength (g)')
    df_clean.drop(columns=['Hardness/firmness/strength (g)'], inplace=True)

    # Split features and labels
    X = df_clean.drop('Hardness/firmness/strength (g)_category', axis=1)
    y = df_clean['Hardness/firmness/strength (g)_category']

    # Save to outputs
    X.to_csv(prepared_data.path, index=False)
    y.to_csv(prepared_labels.path, index=False)
    logger.info("X and y defined and saved to outputs.")
    logger.info("Data preparation complete.")

@component(
    packages_to_install=[
        "pandas==1.5.3",
        "numpy==1.24.4",
        "flaml[automl]==2.3.2",
        "scikit-learn==1.5.2",
        "google-cloud-storage==2.19.0",
        "loguru==0.7.3",
        "joblib==1.3.2"
    ],
    base_image="python:3.10"
)
def train_model(
    prepared_data: Input[Dataset],
    prepared_labels: Input[Dataset],
    metrics: Output[Metrics],
    model_artifact: Output[Model],
    model_name: str,
    model_bucket_name: str,
    project_id: str = "ai-bobby-poc",
):
    import pandas as pd
    from flaml import AutoML
    from loguru import logger
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, f1_score
    from google.cloud import storage
    import joblib

    logger.info("Reading training data...")
    X = pd.read_csv(prepared_data.path)
    y = pd.read_csv(prepared_labels.path).to_numpy()

    # Train and evaluate
    logger.info("Training and evaluating model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    automl_settings = {
        "time_budget": 540,  # total running time in seconds
        "metric": 'log_loss',
        "task": 'classification',
        "estimator_list": ['lgbm', 'rf', 'xgboost', 'extra_tree'],
        "split_type": 'stratified',
        "n_jobs": -1,
        "verbose": 0,
        "seed": 42,
        "early_stop": True
    }

    model = AutoML(**automl_settings)
    model.fit(X_train, y_train, task="classification")
    y_pred = model.predict(X_test)
    logger.info("Model training and evaluation complete.")
    logger.info(f"Model: {model.best_estimator}")
    logger.info(f"Best config: {model.best_config}")

    # Save metrics
    logger.info("Logging metrics...")
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    metrics.log_metric("recall", recall)
    metrics.log_metric("f1_score", f1)
    logger.info("Metrics logged.")

    # Save model
    logger.info("Saving model...")
    model_file_name = f"{model_name}.joblib"
    joblib.dump(model, model_file_name)
    logger.info("Model saved.")

    logger.info(f"Uploading model to GCS bucket: {model_bucket_name}")
    client = storage.Client(project=project_id)
    bucket = client.bucket(model_bucket_name)
    blob = bucket.blob(model_file_name)
    blob.upload_from_filename(model_file_name)
    logger.info(f"Model {model_file_name} uploaded to GCS.")
    # Register model data
    model_artifact.uri = f"gs://{model_bucket_name}/{model_file_name}"
    model_artifact.metadata = {
        "estimator": model.best_estimator,
        "best_config": model.best_config,
        "recall": recall,
        "f1_score": f1
    }
    logger.info("Model training and evaluation complete.")


@dsl.pipeline(
    name='gel-strength-classification-pipeline',
    description='Pipeline to train gel strength classifier.'
)
def classification_pipeline(
    dataset_file_name: str,
    model_save_location: str = "ai-bobby-gel-hardness-models",
):
    from datetime import datetime

    # Generate model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"gel_hardness_classifier_{timestamp}"

    # Prepare data
    prepare_data_task = prepare_data(dataset_file_name=dataset_file_name)

    # Train model
    train_task = train_model(
        prepared_data=prepare_data_task.outputs['prepared_data'],
        prepared_labels=prepare_data_task.outputs['prepared_labels'],
        model_name=model_name,
        model_bucket_name=model_save_location
    )

if __name__ == "__main__":
    from kfp.v2.compiler import Compiler
    
    pipeline_func = classification_pipeline
    package_path = "gel_hardness_classifier_pipeline.yaml"
    
    Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=package_path
    )
