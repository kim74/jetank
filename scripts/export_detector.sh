#!/usr/bin/env bash
# Export an Ultralytics YOLO model to ONNX and build a TensorRT FP16 engine for the Jetson.
#
# Ultralytics is installed into its own venv (CPU torch from PyPI) so it never touches the
# NVIDIA torch build in the runtime venv. Only the resulting engine file is used at runtime.
#
# Usage: scripts/export_detector.sh [model-name] [image-size]
#   e.g. scripts/export_detector.sh yolo11n 640
set -euo pipefail

MODEL="${1:-yolo11n}"
IMGSZ="${2:-640}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/models"
EXPORT_VENV="${JETANK_EXPORT_VENV:-${HOME}/venvs/yolo-export}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${EXPORT_VENV}/bin/python" ]]; then
    echo ">> creating export venv at ${EXPORT_VENV}"
    /usr/bin/python3 -m venv "${EXPORT_VENV}"
    "${EXPORT_VENV}/bin/pip" install --upgrade pip
    "${EXPORT_VENV}/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    "${EXPORT_VENV}/bin/pip" install "ultralytics>=8.3,<9" "onnx>=1.16,<1.18" onnxslim
fi

echo ">> exporting ${MODEL} to ONNX (imgsz=${IMGSZ})"
(
    cd "${OUT_DIR}"
    "${EXPORT_VENV}/bin/yolo" export model="${MODEL}.pt" format=onnx imgsz="${IMGSZ}" \
        opset=17 simplify=True dynamic=False nms=False device=cpu
)

ONNX="${OUT_DIR}/${MODEL}.onnx"
ENGINE="${OUT_DIR}/${MODEL}_${IMGSZ}_fp16.engine"
echo ">> building TensorRT engine ${ENGINE}"
"${TRTEXEC}" --onnx="${ONNX}" --saveEngine="${ENGINE}" --fp16 --memPoolSize=workspace:1024

echo ">> done: ${ENGINE}"
