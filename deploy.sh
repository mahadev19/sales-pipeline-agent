#!/bin/bash
# Submit build to Cloud Build
gcloud builds submit --tag gcr.io/PROJECT_ID/sales-pipeline

# Deploy the image to Cloud Run
gcloud run deploy sales-pipeline-agent \
  --image gcr.io/PROJECT_ID/sales-pipeline \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=your_key_here \
  --memory 1Gi \
  --port 8080
