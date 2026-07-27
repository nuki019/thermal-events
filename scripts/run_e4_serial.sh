#!/bin/bash
# Serial E4: one training at a time to avoid GPU contention
cd "/d/event-camera/DARPA FENCE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/c/ProgramData/anaconda3/python.exe
FLIR="D:/event-camera/DARPA FENCE/data/flir_sted"

for ch in frame event fusion; do
  echo "=== FLIR $ch ==="
  $PY experiments/train_detector.py --data "$FLIR" --channel $ch --epochs 8 --bs 6 --tag _v1 2>&1 | tr -d '\0' | grep -E "params|ep [0-9]|best"
done
echo "=== YOLO anchor ==="
$PY experiments/yolo_baseline.py 2>&1 | tr -d '\0' | grep -iE "mAP|epoch|error" | tail -6
echo E4_SERIAL_DONE
