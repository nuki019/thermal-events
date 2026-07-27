#!/bin/bash
cd "/d/event-camera/DARPA FENCE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/c/ProgramData/anaconda3/python.exe
FLIR="D:/event-camera/DARPA FENCE/data/flir_sted"
STED="D:/event-camera/DARPA FENCE/data/sted"

run() {
  local data=$1 ch=$2 tag=$3 ep=$4 bs=$5 extra=$6
  echo "START $tag $(date +%H:%M:%S)"
  $PY experiments/train_detector.py --data "$data" --channel $ch --epochs $ep --bs $bs --tag $tag $extra > "experiments/results/log_${tag}.txt" 2>&1
  echo "END $tag $(date +%H:%M:%S)"
}

run "$FLIR" frame _f1 10 6
run "$FLIR" event _f1 10 6
run "$FLIR" fusion _f1 10 6
run "$STED" frame _s1 8 6
run "$STED" event _s1 8 6
run "$STED" fusion _s1 8 6
run "$STED" event _sr1 8 4 "--recurrent --chunk 8"
echo FINAL_TRAININGS_DONE
