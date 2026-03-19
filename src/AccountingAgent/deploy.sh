#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
REGION="${GCP_REGION:-europe-north1}"
SERVICE_NAME="${SERVICE_NAME:-tripletex-agent}"
LLM_PROVIDER="${LLM_PROVIDER:-openai}"

echo "=== Building and deploying $SERVICE_NAME ==="
echo "  Project:  $PROJECT_ID"
echo "  Region:   $REGION"
echo "  Provider: $LLM_PROVIDER"
echo ""

gcloud run deploy "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --source . \
  --region "$REGION" \
  --platform managed \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 5 \
  --concurrency 3 \
  --allow-unauthenticated \
  --set-env-vars "LLM_PROVIDER=$LLM_PROVIDER,LOG_LEVEL=INFO" \
  --update-secrets "OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest"

echo ""
echo "=== Deployment complete ==="
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" --format 'value(status.url)')
echo "Service URL: $SERVICE_URL"
echo "Submit this URL at: https://app.ainm.no/submit/tripletex"
echo "  Endpoint: ${SERVICE_URL}/solve"
