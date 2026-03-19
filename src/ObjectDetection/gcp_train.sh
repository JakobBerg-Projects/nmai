#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# GCP Remote GPU Training
# Usage:
#   ./gcp_train.sh --strategy detector --model yolov8x.pt --epochs 100
#   ./gcp_train.sh --strategy classifier --epochs 30
#   ./gcp_train.sh --strategy two_stage   # runs detector + crops + classifier
#   ./gcp_train.sh --strategy multiclass --model yolov8x.pt --epochs 100
#   ./gcp_train.sh --delete               # tear down the VM
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────
GPU_TYPE="t4"
MODEL="yolov8x.pt"
EPOCHS=100
IMGSZ=640
STRATEGY="detector"
CLS_EPOCHS=30
DELETE_ONLY=false

ZONE="europe-west1-b"
VM_NAME="yolo-training"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)        GPU_TYPE="$2"; shift 2 ;;
    --model)      MODEL="$2"; shift 2 ;;
    --epochs)     EPOCHS="$2"; shift 2 ;;
    --imgsz)      IMGSZ="$2"; shift 2 ;;
    --strategy)   STRATEGY="$2"; shift 2 ;;
    --cls-epochs) CLS_EPOCHS="$2"; shift 2 ;;
    --zone)       ZONE="$2"; shift 2 ;;
    --delete)     DELETE_ONLY=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── GPU config mapping ──────────────────────────────────────────────
case "$GPU_TYPE" in
  t4)
    ACCELERATOR="type=nvidia-tesla-t4,count=1"
    MACHINE_TYPE="n1-standard-8"
    ;;
  l4)
    ACCELERATOR="type=nvidia-l4,count=1"
    MACHINE_TYPE="g2-standard-8"
    ;;
  *)
    echo "Unsupported GPU: $GPU_TYPE (use t4 or l4)"
    exit 1
    ;;
esac

# ── Delete mode ─────────────────────────────────────────────────────
if $DELETE_ONLY; then
  echo "Deleting VM '$VM_NAME'..."
  gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet 2>/dev/null || echo "VM not found."
  exit 0
fi

echo "══════════════════════════════════════════════════════════"
echo "  GCP Remote Training"
echo "  Strategy: $STRATEGY"
echo "  GPU:      $GPU_TYPE ($ACCELERATOR)"
echo "  Model:    $MODEL"
echo "  Epochs:   $EPOCHS  |  ImgSz: $IMGSZ"
echo "  Zone:     $ZONE"
echo "══════════════════════════════════════════════════════════"

# ── Step 1: Create VM if it doesn't exist ───────────────────────────
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
  echo ">> VM '$VM_NAME' already exists, reusing."
  gcloud compute instances start "$VM_NAME" --zone="$ZONE" 2>/dev/null || true
else
  echo ">> Creating VM '$VM_NAME' with $GPU_TYPE GPU..."
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
fi

# Wait for SSH to be ready
echo ">> Waiting for SSH..."
for i in $(seq 1 30); do
  if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo ok" &>/dev/null; then
    echo "   SSH ready."
    break
  fi
  sleep 10
  echo "   retry $i..."
done

# Resolve remote home directory
REMOTE_HOME="$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo \$HOME" 2>/dev/null)"
REMOTE_DIR="${REMOTE_HOME}/training"
echo ">> Remote dir: $REMOTE_DIR"

# ── Step 2: Upload code and data ────────────────────────────────────
echo ">> Uploading code and data..."

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="mkdir -p ${REMOTE_DIR}/{data/train,strategies,training,weights}"

# Upload all python files and modules
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
  "$VM_NAME:${REMOTE_DIR}/training/" --zone="$ZONE"

# Upload annotations
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

# Upload product reference images for classifier training
if [ "$STRATEGY" = "classifier" ] || [ "$STRATEGY" = "two_stage" ]; then
  REF_COUNT=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="ls ${REMOTE_DIR}/data/NM_NGD_product_images/ 2>/dev/null | wc -l" 2>/dev/null || echo "0")
  if [ "$REF_COUNT" -lt 300 ]; then
    echo ">> Uploading product reference images..."
    gcloud compute scp --recurse \
      "$SCRIPT_DIR/data/NM_NGD_product_images" \
      "$VM_NAME:${REMOTE_DIR}/data/" --zone="$ZONE"
  else
    echo ">> Reference images already on VM, skipping."
  fi
fi

# ── Step 3: Install dependencies and run training ───────────────────
echo ">> Installing dependencies and starting training..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- bash -lc "
  set -e
  cd ${REMOTE_DIR}

  # Install system dependency for OpenCV
  sudo apt-get update -qq && sudo apt-get install -y -qq libgl1-mesa-glx libglib2.0-0

  # Install requirements (use VM's pre-installed torch, don't override)
  pip install -q 'ultralytics>=8.3.0' opencv-python-headless pillow

  # Verify GPU
  python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"WARNING: No GPU!\")'

  case '$STRATEGY' in
    detector)
      echo '=== Training single-class detector ==='
      python3 -m training.train_detector --mode single --model $MODEL --epochs $EPOCHS --imgsz $IMGSZ
      ;;
    multiclass)
      echo '=== Training multiclass detector ==='
      python3 -m training.train_detector --mode multi --model $MODEL --epochs $EPOCHS --imgsz $IMGSZ
      ;;
    classifier)
      echo '=== Extracting crops ==='
      python3 -m training.extract_crops
      echo '=== Training classifier ==='
      python3 -m training.train_classifier --epochs $CLS_EPOCHS
      ;;
    two_stage)
      echo '=== Phase 1: Training single-class detector ==='
      python3 -m training.train_detector --mode single --model $MODEL --epochs $EPOCHS --imgsz $IMGSZ
      echo '=== Phase 2: Extracting crops ==='
      python3 -m training.extract_crops
      echo '=== Phase 3: Training classifier ==='
      python3 -m training.train_classifier --epochs $CLS_EPOCHS
      ;;
    *)
      echo 'Unknown strategy: $STRATEGY'
      exit 1
      ;;
  esac

  echo 'Training complete!'
"

# ── Step 4: Download weights ─────────────────────────────────────────
echo ">> Downloading trained weights..."
mkdir -p "$SCRIPT_DIR/weights"

case "$STRATEGY" in
  detector)
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/detector.pt" \
      "$SCRIPT_DIR/weights/detector.pt" --zone="$ZONE"
    ;;
  multiclass)
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/best.pt" \
      "$SCRIPT_DIR/weights/best.pt" --zone="$ZONE"
    ;;
  classifier)
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/classifier.pt" \
      "$SCRIPT_DIR/weights/classifier.pt" --zone="$ZONE"
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/ref_embeddings.pt" \
      "$SCRIPT_DIR/weights/ref_embeddings.pt" --zone="$ZONE"
    ;;
  two_stage)
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/detector.pt" \
      "$SCRIPT_DIR/weights/detector.pt" --zone="$ZONE"
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/classifier.pt" \
      "$SCRIPT_DIR/weights/classifier.pt" --zone="$ZONE"
    gcloud compute scp \
      "$VM_NAME:${REMOTE_DIR}/weights/ref_embeddings.pt" \
      "$SCRIPT_DIR/weights/ref_embeddings.pt" --zone="$ZONE"
    ;;
esac

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Training complete!"
echo "  Strategy: $STRATEGY"
echo "  Weights:  $SCRIPT_DIR/weights/"
echo "══════════════════════════════════════════════════════════"

# ── Step 5: Stop VM to save costs ───────────────────────────────────
echo ">> Stopping VM to avoid charges..."
gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
echo "VM stopped. Run './gcp_train.sh --delete' to fully remove it."
