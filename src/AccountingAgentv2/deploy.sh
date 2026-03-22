#!/bin/bash
set -e

echo "Deploying nmai-agent to Cloud Run..."

gcloud run deploy nmai-agent \
  --source=. \
  --region=europe-north1 \
  --platform=managed \
  --allow-unauthenticated \
  --timeout=300 \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=1 \
  --max-instances=10 \
  --set-secrets=ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest \
  --project=ai-nm26osl-1791 \
  --quiet

echo ""
echo "Live at: https://nmai-agent-609347898393.europe-north1.run.app/solve"
