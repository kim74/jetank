#!/usr/bin/env bash
# Train the toy + floor segmentation model (YOLO11n-seg) on the Jetson's GPU.
#
# Ultralytics runs in its own venv with NVIDIA-built torch/torchvision for JetPack 6 (CUDA
# 12.6), so it never touches the robot's runtime venv. The first run creates that venv
# (downloads several GB) and Ultralytics fetches the pretrained weights from GitHub.
#
# Stop `jetank teleop` first: training needs most of the 8 GB the CPU and GPU share. CUDA on
# the Jetson only gets memory that is really free, not the file cache ("NvMapMemAlloc...
# error 12"): if it runs out, use a smaller batch or free the cache first with
#   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Training can take a while; run it under nohup or tmux so a dropped SSH session doesn't end it.
#
# Usage: scripts/train_detector.sh [epochs] [batch] [image-size]
#   e.g. scripts/train_detector.sh 100 4 640
# Result: models/toys-seg.pt (best epoch), plus plots and metrics in models/runs/toys-seg*/
set -euo pipefail

EPOCHS="${1:-100}"
BATCH="${2:-4}"
IMGSZ="${3:-640}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${JETANK_DATASET:-${REPO_ROOT}/data/dataset}/data.yaml"
OUT_DIR="${REPO_ROOT}/models"
TRAIN_VENV="${JETANK_TRAIN_VENV:-${HOME}/venvs/yolo-train}"
INDEX="${JETANK_TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
TORCH="torch==2.8.0"
TORCHVISION="torchvision==0.23.0"

if [[ ! -f "${DATA}" ]]; then
    echo "no dataset at ${DATA}: run \`jetank dataset build\` first" >&2
    exit 1
fi

if [[ ! -x "${TRAIN_VENV}/bin/python" ]]; then
    echo ">> creating training venv at ${TRAIN_VENV}"
    /usr/bin/python3 -m venv "${TRAIN_VENV}"
    "${TRAIN_VENV}/bin/pip" install --upgrade pip
    "${TRAIN_VENV}/bin/pip" install "${TORCH}" "${TORCHVISION}" "numpy<2" --index-url "${INDEX}" \
        --extra-index-url https://pypi.org/simple
    # Pin torch so installing Ultralytics cannot swap it for a CPU build from PyPI, and keep
    # NumPy 1.x: the Jetson torch wheels are built against it (OpenCV >= 4.11 needs NumPy 2).
    printf '%s\n' "${TORCH}" "${TORCHVISION}" "numpy<2" "opencv-python<4.11" \
        > "${TRAIN_VENV}/constraints.txt"
    "${TRAIN_VENV}/bin/pip" install "ultralytics>=8.3,<9" -c "${TRAIN_VENV}/constraints.txt"
fi

"${TRAIN_VENV}/bin/python" - <<'EOF'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(f"torch {torch.__version__} sees no GPU: training on the CPU would take days")
print(f">> torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
EOF

mkdir -p "${OUT_DIR}/runs"
echo ">> training YOLO11n-seg on ${DATA}: ${EPOCHS} epochs, batch ${BATCH}, ${IMGSZ} px"
cd "${OUT_DIR}"  # the pretrained weights are downloaded here
# fliplr: a mirrored room is still a room. workers=2: the dataloader shares the CPU and RAM.
"${TRAIN_VENV}/bin/yolo" segment train \
    model=yolo11n-seg.pt data="${DATA}" epochs="${EPOCHS}" batch="${BATCH}" imgsz="${IMGSZ}" \
    device=0 workers=2 patience=30 fliplr=0.5 \
    project="${OUT_DIR}/runs" name=toys-seg

BEST="$(ls -td "${OUT_DIR}"/runs/toys-seg*/ | head -1)weights/best.pt"
cp "${BEST}" "${OUT_DIR}/toys-seg.pt"
echo ">> done: ${OUT_DIR}/toys-seg.pt (from ${BEST})"
echo ">> next: scripts/export_detector.sh ${OUT_DIR}/toys-seg.pt ${IMGSZ}"
