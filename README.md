# AI Bobby Gel Hardness Classification PoC

This repository contains the code for the AI Bobby Gel Hardness Classification PoC.

## 1 - Folders and main files

- `src/app/main.py`: Streamlit app for making predictions with the trained models.
- `src/compile_pipeline.py`: Script to compile the Vertex AI pipeline.
- `notebooks/`: Folder containing the notebooks used for the PoC.
- `data/`: Folder containing the datasets used for the PoC.
- `gels_hardness_classification_pipeline.yaml`: Vertex AI pipeline definition.

## 2 - Where to find the Vertex AI pipeline definition, datasets and trained models.

Note: You must be given access to the GCP project `ai-bobby-poc` to access the datasets and trained models.

- Pipeline definition: [pipeline file in this repository](gel_hardness_classification_pipeline.yaml) or [pipeline file in GCS](https://console.cloud.google.com/vertex-ai/pipelines/ai-bobby-gel-hardness-pipeline-definitions).
- Datasets: [GCS bucket](https://console.cloud.google.com/storage/browser/ai-bobby-gel-hardness-datasets)
- Trained models: [GCS bucket](https://console.cloud.google.com/storage/browser/ai-bobby-gel-hardness-models)

## 3 - Building and deploying the prediction app

Every commit to the `main` branch will trigger a build and push of the Docker image and deploy the Cloud Run service following the [cloudbuild.yml](cloudbuild.yml) definition.

## 4 - How to re-compile the pipeline

1. Install the uv package manager

```bash
pip install uv
```

2. Install the dependencies

```bash
uv sync
```

3. Compile the pipeline

```bash
uv run python src/compile_pipeline.py
```

The file will be compiled and saved as `gel_hardness_classifier_pipeline.yaml`.

