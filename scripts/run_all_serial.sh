#!/bin/bash
# Strictly serial experiment pipeline
cd "/d/event-camera/DARPA FENCE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/c/ProgramData/anaconda3/python.exe
FLIR="D:/event-camera/DARPA FENCE/data/flir_sted"
STED="D:/event-camera/DARPA FENCE/data/sted"

echo "##### 1. FLIR frame #####"
$PY experiments/train_detector.py --data "$FLIR" --channel frame --epochs 10 --bs 6 --tag _v3 2>&1 | tr -d '\0' | grep -E "ep [0-9]|best"
echo "##### 2. FLIR event #####"
$PY experiments/train_detector.py --data "$FLIR" --channel event --epochs 10 --bs 6 --tag _v3 2>&1 | tr -d '\0' | grep -E "ep [0-9]|best"
echo "##### 3. FLIR fusion #####"
$PY experiments/train_detector.py --data "$FLIR" --channel fusion --epochs 10 --bs 6 --tag _v3 2>&1 | tr -d '\0' | grep -E "ep [0-9]|best"
echo "##### 4. YOLO anchor #####"
$PY experiments/yolo_baseline.py 2>&1 | tr -d '\0' | grep -iE "mAP50|mAP|all.*0\." | tail -4
echo "##### 5. STED frame #####"
$PY experiments/train_detector.py --data "$STED" --channel frame --epochs 8 --bs 6 --tag _v3 2>&1 | tr -d '\0' | grep -E "ep [0-9]|best"
echo "##### 6. STED event #####"
$PY experiments/train_detector.py --data "$STED" --channel event --epochs 8 --bs 6 --tag _v3 2>&1 | tr -d '\0' | grep -E "ep [0-9]|best"
echo "##### 7. STED fusion #####"
$PY experiments/train_detector.py --data "$STED" --channel fusion --epochs 8 --bs 6 --tag _v3 2>&1 | tr -d '\0' | grep -E "ep [0-9]|best"
echo "##### 8. STED event recurrent #####"
$PY experiments/train_detector.py --data "$STED" --channel event --epochs 8 --bs 4 --recurrent --chunk 8 --tag _v3 2>&1 | tr -d '\0' | grep -E "ep [0-9]|best"
echo ALL_SERIAL_DONE
