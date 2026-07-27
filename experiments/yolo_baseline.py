"""YOLOv8n frame-channel baseline on FLIR thermal (external anchor)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    model = YOLO('yolov8n.pt')
    model.train(data=os.path.join(ROOT, 'data', 'flir_yolo', 'flir_thermal.yaml'),
                epochs=12, imgsz=640, batch=16, device=0, workers=0,
                project=os.path.join(ROOT, 'experiments', 'results'),
                name='yolov8n_flir', exist_ok=True, verbose=False)
    r = model.val()
    print('mAP50-95:', round(float(r.box.map), 4), 'mAP50:', round(float(r.box.map50), 4))

if __name__ == '__main__':
    main()
