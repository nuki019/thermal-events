#!/bin/bash
# E4 training matrix: run sequentially to avoid GPU contention
set -e
cd "/d/event-camera/DARPA FENCE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/c/ProgramData/anaconda3/python.exe
DATA="D:/event-camera/DARPA FENCE/data/sted"
FLIR="D:/event-camera/DARPA FENCE/data/flir_sted"

# STED synthetic channels
$PY experiments/train_detector.py --data "$DATA" --channel frame  --epochs 8 --bs 6 --tag _v1
$PY experiments/train_detector.py --data "$DATA" --channel event  --epochs 8 --bs 6 --tag _v1
$PY experiments/train_detector.py --data "$DATA" --channel fusion --epochs 8 --bs 6 --tag _v1
$PY experiments/train_detector.py --data "$DATA" --channel event  --epochs 8 --bs 4 --recurrent --chunk 8 --tag _v1

# FLIR-STED real channels
$PY experiments/train_detector.py --data "$FLIR" --channel frame  --epochs 8 --bs 6 --tag _v1
$PY experiments/train_detector.py --data "$FLIR" --channel event  --epochs 8 --bs 6 --tag _v1
$PY experiments/train_detector.py --data "$FLIR" --channel fusion --epochs 8 --bs 6 --tag _v1
echo E4_ALL_DONE
