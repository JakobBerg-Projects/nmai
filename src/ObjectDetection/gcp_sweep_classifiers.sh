#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# GCP Single Classifier Training on a specific VM
#
# Usage (one per terminal):
#   ./gcp_sweep_classifiers.sh --vm cls-sweep-20260322-012722 --arch convnext_small --crop 224 --batch 48 --lr 5e-4
#   ./gcp_sweep_classifiers.sh --vm nmai-a100 --arch swin_v2_s --crop 256 --batch 32 --lr 5e-4
#   ./gcp_sweep_classifiers.sh --vm cls-swinv2t --new --arch swin_v2_t --crop 256 --batch 48 --lr 5e-4
#   ./gcp_sweep_classifiers.sh --vm cls-effv2m --new --arch efficientnet_v2_m --crop 384 --batch 24 --lr 5e-4
#   ./gcp_sweep_classifiers.sh --cleanup   # delete all cls-* VMs
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

ZONE="us-central1-f"
ACCELERATOR="type=nvidia-tesla-a100,count=1"
MACHINE_TYPE="a2-highgpu-1g"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CLS_EPOCHS=60
PATIENCE=15
DET_IMGSZ=1280
BEST_SCORE_FILE="$SCRIPT_DIR/weights/best_score"
BASELINE_SCORE=$(cat "$BEST_SCORE_FILE" 2>/dev/null || echo "0.2627")

VM_NAME=""
ARCH=""
CROP=""
BATCH=""
LR="5e-4"
CREATE_NEW=false

# ── Parse args ────────────────────────────────────────────────────────
if [[ "${1:-}" == "--cleanup" ]]; then
  echo "Deleting all cls-* VMs in $ZONE..."
  for vm in $(gcloud compute instances list --filter="name~^cls-" --zones="$ZONE" --format="value(name)" 2>/dev/null); do
    echo "  Deleting $vm..."
    gcloud compute instances delete "$vm" --zone="$ZONE" --quiet &
  done
  wait
  echo "Done."
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vm)       VM_NAME="$2"; shift 2 ;;
    --arch)     ARCH="$2"; shift 2 ;;
    --crop)     CROP="$2"; shift 2 ;;
    --batch)    BATCH="$2"; shift 2 ;;
    --lr)       LR="$2"; shift 2 ;;
    --epochs)   CLS_EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --baseline) BASELINE_SCORE="$2"; shift 2 ;;
    --new)      CREATE_NEW=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$VM_NAME" || -z "$ARCH" || -z "$CROP" || -z "$BATCH" ]]; then
  echo "Required: --vm NAME --arch ARCH --crop SIZE --batch SIZE"
  echo "Optional: --lr LR --new (create VM) --baseline SCORE --epochs N --patience N"
  exit 1
fi

echo "══════════════════════════════════════════════════════════"
echo "  VM:       $VM_NAME $(if $CREATE_NEW; then echo '(NEW)'; else echo '(existing)'; fi)"
echo "  Arch:     $ARCH"
echo "  Crop:     $CROP  |  Batch: $BATCH  |  LR: $LR"
echo "  Epochs:   $CLS_EPOCHS  |  Patience: $PATIENCE"
echo "  Baseline: $BASELINE_SCORE (must beat to save)"
echo "══════════════════════════════════════════════════════════"

# ── Create or start VM ───────────────────────────────────────────────
if $CREATE_NEW; then
  if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo "ERROR: VM '$VM_NAME' already exists. Pick a different name or drop --new."
    exit 1
  fi
  echo ">> Creating VM '$VM_NAME'..."
  gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="$ACCELERATOR" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=200GB \
    --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=True" \
    --scopes=default
else
  if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
    echo "ERROR: VM '$VM_NAME' not found. Use --new to create it."
    exit 1
  fi
  echo ">> Starting VM '$VM_NAME' (if stopped)..."
  gcloud compute instances start "$VM_NAME" --zone="$ZONE" 2>/dev/null || true
fi

# Wait for SSH
echo ">> Waiting for SSH..."
for i in $(seq 1 30); do
  if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo ok" &>/dev/null; then
    echo "   SSH ready."
    break
  fi
  sleep 10
  echo "   retry $i..."
done

REMOTE_HOME="$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo \$HOME" 2>/dev/null)"
REMOTE_DIR="${REMOTE_HOME}/training"

# ── Upload code and data ─────────────────────────────────────────────
echo ">> Uploading code and data..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="mkdir -p ${REMOTE_DIR}/{data/train,strategies,training,weights}"

gcloud compute scp \
  "$SCRIPT_DIR/run.py" \
  "$SCRIPT_DIR/prepare_dataset.py" \
  "$SCRIPT_DIR/train.py" \
  "$VM_NAME:${REMOTE_DIR}/" --zone="$ZONE"

gcloud compute scp \
  "$SCRIPT_DIR/strategies/__init__.py" \
  "$SCRIPT_DIR/strategies/base.py" \
  "$SCRIPT_DIR/strategies/yolo_multiclass.py" \
  "$SCRIPT_DIR/strategies/detection_only.py" \
  "$SCRIPT_DIR/strategies/two_stage.py" \
  "$VM_NAME:${REMOTE_DIR}/strategies/" --zone="$ZONE"

gcloud compute scp \
  "$SCRIPT_DIR/training/__init__.py" \
  "$SCRIPT_DIR/training/train_detector.py" \
  "$SCRIPT_DIR/training/train_classifier.py" \
  "$SCRIPT_DIR/training/extract_crops.py" \
  "$SCRIPT_DIR/training/validate_pipeline.py" \
  "$VM_NAME:${REMOTE_DIR}/training/" --zone="$ZONE"

gcloud compute scp \
  "$SCRIPT_DIR/data/train/annotations.json" \
  "$VM_NAME:${REMOTE_DIR}/data/train/annotations.json" --zone="$ZONE"

# Upload images only if not already present
IMAGE_COUNT=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="ls ${REMOTE_DIR}/data/train/images/ 2>/dev/null | wc -l" 2>/dev/null || echo "0")
if [ "$IMAGE_COUNT" -lt 248 ]; then
  echo ">> Uploading training images..."
  gcloud compute scp --recurse \
    "$SCRIPT_DIR/data/train" \
    "$VM_NAME:${REMOTE_DIR}/data/" --zone="$ZONE"
else
  echo ">> Training images already on VM ($IMAGE_COUNT images), skipping."
fi

# Upload product reference images
REF_COUNT=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="ls ${REMOTE_DIR}/data/NM_NGD_product_images/ 2>/dev/null | wc -l" 2>/dev/null || echo "0")
if [ "$REF_COUNT" -lt 300 ] && [ -d "$SCRIPT_DIR/data/NM_NGD_product_images" ]; then
  echo ">> Uploading product reference images..."
  gcloud compute scp --recurse \
    "$SCRIPT_DIR/data/NM_NGD_product_images" \
    "$VM_NAME:${REMOTE_DIR}/data/" --zone="$ZONE"
else
  echo ">> Reference images already on VM, skipping."
fi

# Upload detector weights and marker
if [ -f "$SCRIPT_DIR/weights/detector.pt" ]; then
  gcloud compute scp "$SCRIPT_DIR/weights/detector.pt" \
    "$VM_NAME:${REMOTE_DIR}/weights/detector.pt" --zone="$ZONE" 2>/dev/null && \
    echo ">> Uploaded detector.pt" || true
fi
if [ -f "$SCRIPT_DIR/weights/detector_updated" ]; then
  gcloud compute scp "$SCRIPT_DIR/weights/detector_updated" \
    "$VM_NAME:${REMOTE_DIR}/weights/detector_updated" --zone="$ZONE" 2>/dev/null && \
    echo ">> Uploaded detector_updated marker" || true
fi

# ── Train ─────────────────────────────────────────────────────────────
echo ">> Starting training..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
set -e
cd ${REMOTE_DIR}
sudo apt-get update -qq && sudo apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 2>&1 | tail -1
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q ultralytics==8.1.0 opencv-python-headless pillow onnxruntime pycocotools
python3 -c 'import torch; print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO GPU\")'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m training.extract_crops

python3 -m training.train_classifier \
  --arch ${ARCH} \
  --crop-size ${CROP} \
  --batch-size ${BATCH} \
  --lr ${LR} \
  --epochs ${CLS_EPOCHS} \
  --patience ${PATIENCE} \
  --detector-path ${REMOTE_DIR}/weights/detector.pt \
  --det-imgsz ${DET_IMGSZ} \
  --best-score ${BASELINE_SCORE}
"

# ── Download if improved ──────────────────────────────────────────────
SCORE=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
  --command="cat ${REMOTE_DIR}/weights/classifier_updated 2>/dev/null" 2>/dev/null || echo "")

if [ -n "$SCORE" ]; then
  # Only download if this score actually beats the current local best
  CURRENT_BEST=$(cat "$BEST_SCORE_FILE" 2>/dev/null || echo "0.0")
  if python3 -c "exit(0 if float('$SCORE') > float('$CURRENT_BEST') else 1)"; then
    echo ""
    echo ">> IMPROVED! score=$SCORE (was $CURRENT_BEST) — downloading weights..."
    mkdir -p "$SCRIPT_DIR/weights"
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/classifier.pt" \
      "$SCRIPT_DIR/weights/classifier.pt" --zone="$ZONE"
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/ref_embeddings.pt" \
      "$SCRIPT_DIR/weights/ref_embeddings.pt" --zone="$ZONE"
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/config.json" \
      "$SCRIPT_DIR/weights/config.json" --zone="$ZONE" 2>/dev/null || true
    echo "$SCORE" > "$BEST_SCORE_FILE"
    echo ">> Updated best_score to $SCORE"
  else
    echo ""
    echo ">> Score $SCORE did NOT beat current local best ($CURRENT_BEST) — skipping download."
  fi
else
  echo ""
  echo ">> No improvement over baseline ($BASELINE_SCORE) — weights NOT overwritten."
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Done! VM '$VM_NAME' is still running."
echo "  Stop:   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "  Delete: gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet"
echo "══════════════════════════════════════════════════════════"
