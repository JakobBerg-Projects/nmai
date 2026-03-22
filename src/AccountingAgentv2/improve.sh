#!/bin/bash
set -e

echo "=== NMiAI Agent Feedback Loop ==="
echo ""

# 1. Fetch logs
echo "Fetching Cloud Run logs..."
gcloud run services logs read nmai-agent \
  --region=europe-north1 \
  --limit=300 \
  --project=ai-nm26osl-1791 \
  > /tmp/nmai-logs.txt 2>/dev/null

LINE_COUNT=$(wc -l < /tmp/nmai-logs.txt)
echo "Got $LINE_COUNT log lines"
echo ""

# 2. Analyze (now includes performance tracking with streak/trends)
echo "Analyzing with performance tracking..."
python3 analyze.py /tmp/nmai-logs.txt

echo ""
echo "=== Suggestions written to suggestions.md ==="
echo ""

# 3. Show trend summary if history exists
if [ -f "performance_history.json" ]; then
    echo "--- Performance Tracker ---"
    python3 -c "from performance_tracker import get_trend_summary, should_revert; print(get_trend_summary()); print(); print('!! REVERT ANBEFALT !!' if should_revert() else 'Ingen revert nødvendig.')"
    echo ""
fi

echo "To apply suggestions:"
echo "  1. Read suggestions.md"
echo "  2. Apply changes manually or with: claude 'Apply the suggestions in suggestions.md'"
echo "  3. Deploy with: bash deploy.sh"
echo ""
echo "Performance history saved to performance_history.json"
