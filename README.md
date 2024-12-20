# AI Bobby Gel Hardness Classification PoC

This repository contains the code for the AI Bobby Gel Hardness Classification PoC.

## Folders and main files

- `src/app/main.py`: Streamlit app for making predictions with the trained models.
- `src/compile_pipeline.py`: Script to compile the Vertex AI pipeline.
- `notebooks/`: Folder containing the notebooks used for the PoC.
- `data/`: Folder containing the datasets used for the PoC.
- `gels_hardness_classification_pipeline.yaml`: Vertex AI pipeline definition.

## Where to find the Vertex AI pipeline definition, datasets and trained models.

Note: You must be given access to the GCP project `ai-bobby-poc` to access the datasets and trained models.

- Pipeline definition: [pipeline file in this repository](gel_hardness_classification_pipeline.yaml) or [pipeline file in GCS](https://console.cloud.google.com/vertex-ai/pipelines/ai-bobby-gel-hardness-pipeline-definitions).
- Datasets: [GCS bucket](https://console.cloud.google.com/storage/browser/ai-bobby-gel-hardness-datasets)
- Trained models: [GCS bucket](https://console.cloud.google.com/storage/browser/ai-bobby-gel-hardness-models)

## How to re-compile the pipeline

1. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

2. Compile the pipeline

```bash
python src/compile_pipeline.py
```

## Building, pushing and deploying the prediction app

### Prerequisites

- `docker` installed.
- `gcloud` CLI authenticated with the `ai-bobby-poc` project.

to authenticate with the `ai-bobby-poc` project, run:

```bash
gcloud auth login
```

### Build and push the Docker image

```bash
docker build -t us-central1-docker.pkg.dev/ai-bobby-poc/cloud-run-source-deploy/gel-hardness-prediction -f Dockerfile .
```

```bash
docker push us-central1-docker.pkg.dev/ai-bobby-poc/cloud-run-source-deploy/gel-hardness-prediction
```

### Deploy the Cloud Run service

```bash
gcloud run deploy gel-hardness-prediction --image us-central1-docker.pkg.dev/ai-bobby-poc/cloud-run-source-deploy/gel-hardness-prediction --platform managed --region us-central1 --cpu=4 --memory=4Gi --min-instances=0 --max-instances=2 --concurrency=80 --timeout=300 --port=8080
```
