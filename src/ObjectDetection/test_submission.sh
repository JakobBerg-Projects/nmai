#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Test submission compatibility with sandbox environment
# Creates a venv with ultralytics==8.1.0 and verifies weights load
#
# Usage: ./test_submission.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.test_venv"
TEST_IMG_DIR="$SCRIPT_DIR/data/train/images"
TEST_OUTPUT="$SCRIPT_DIR/test_predictions.json"

echo "══════════════════════════════════════════════════════════"
echo "  Submission Compatibility Test"
echo "  Simulating sandbox: ultralytics==8.1.0 + torch 2.6"
echo "══════════════════════════════════════════════════════════"

# Create isolated venv
if [ ! -d "$VENV_DIR" ]; then
  echo ">> Creating test venv..."
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo ">> Installing sandbox-equivalent packages..."
pip install -q torch==2.6.0 torchvision==0.21.0 ultralytics==8.1.0 opencv-python-headless pillow

# Check what weights exist
echo ""
echo ">> Checking weights..."
if [ -d "$SCRIPT_DIR/weights" ]; then
  ls -lh "$SCRIPT_DIR/weights/"
  TOTAL_SIZE=$(du -sh "$SCRIPT_DIR/weights/" | cut -f1)
  echo "   Total weight size: $TOTAL_SIZE (must be < 420 MB)"
else
  echo "   ERROR: No weights/ directory found"
  exit 1
fi

# Test loading
echo ""
echo ">> Testing weight loading with ultralytics==8.1.0..."
python3 -c "
from pathlib import Path
import sys

weights_dir = Path('$SCRIPT_DIR/weights')

# Test detector
det = weights_dir / 'detector.pt'
if det.exists():
    try:
        from ultralytics import YOLO
        model = YOLO(str(det))
        print(f'  OK  detector.pt loaded ({model.task}, nc={len(model.names)})')
    except Exception as e:
        print(f'  ERR detector.pt: {e}')
        sys.exit(1)

# Test classifier
cls = weights_dir / 'classifier.pt'
if cls.exists():
    try:
        import torch
        from torchvision.models import mobilenet_v3_small
        model = mobilenet_v3_small(weights=None)
        model.classifier = torch.nn.Identity()
        state = torch.load(str(cls), map_location='cpu')
        model.load_state_dict(state)
        print(f'  OK  classifier.pt loaded')
    except Exception as e:
        print(f'  ERR classifier.pt: {e}')
        sys.exit(1)

# Test ref embeddings
ref = weights_dir / 'ref_embeddings.pt'
if ref.exists():
    try:
        import torch
        refs = torch.load(str(ref), map_location='cpu')
        print(f'  OK  ref_embeddings.pt loaded ({len(refs)} categories)')
    except Exception as e:
        print(f'  ERR ref_embeddings.pt: {e}')
        sys.exit(1)

best = weights_dir / 'best.pt'
if best.exists():
    try:
        from ultralytics import YOLO
        model = YOLO(str(best))
        print(f'  OK  best.pt loaded ({model.task}, nc={len(model.names)})')
    except Exception as e:
        print(f'  ERR best.pt: {e}')
        sys.exit(1)
"

# Run inference on a few test images
echo ""
echo ">> Running inference test..."
# Pick 3 images
TEST_IMGS=$(ls "$TEST_IMG_DIR"/*.jpg 2>/dev/null | head -3)
if [ -z "$TEST_IMGS" ]; then
  echo "   No test images found in $TEST_IMG_DIR"
  exit 1
fi

# Create temp dir with just 3 images
TEMP_DIR=$(mktemp -d)
for img in $TEST_IMGS; do
  cp "$img" "$TEMP_DIR/"
done

cd "$SCRIPT_DIR"
python3 run.py --input "$TEMP_DIR" --output "$TEST_OUTPUT"

# Validate output format
echo ""
echo ">> Validating output format..."
python3 -c "
import json

with open('$TEST_OUTPUT') as f:
    preds = json.load(f)

print(f'   Predictions: {len(preds)}')

if not preds:
    print('   WARNING: No predictions generated!')
else:
    p = preds[0]
    required = {'image_id', 'category_id', 'bbox', 'score'}
    actual = set(p.keys())
    missing = required - actual
    extra = actual - required

    if missing:
        print(f'   ERROR: Missing fields: {missing}')
    else:
        print(f'   OK  All required fields present')

    if extra:
        print(f'   WARNING: Extra fields: {extra}')

    # Check types
    assert isinstance(p['image_id'], int), f'image_id must be int, got {type(p[\"image_id\"])}'
    assert isinstance(p['category_id'], int), f'category_id must be int, got {type(p[\"category_id\"])}'
    assert isinstance(p['bbox'], list) and len(p['bbox']) == 4, f'bbox must be [x,y,w,h]'
    assert isinstance(p['score'], float), f'score must be float, got {type(p[\"score\"])}'
    assert 0 <= p['category_id'] <= 356, f'category_id must be 0-356, got {p[\"category_id\"]}'

    print(f'   OK  Field types correct')
    print(f'   Example: {json.dumps(p)}')
"

# Cleanup
rm -rf "$TEMP_DIR" "$TEST_OUTPUT"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  All checks passed!"
echo "══════════════════════════════════════════════════════════"

deactivate
