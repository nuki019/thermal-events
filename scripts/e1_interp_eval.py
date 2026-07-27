"""E1: interpolation evaluation on real HIT-UAV thermal videos + synthetic."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from thermal_events.pipeline.interp import eval_interpolation


def load_video_gray(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames) if frames else np.empty((0,))


if __name__ == '__main__':
    root = r'D:/event-camera/DARPA FENCE'
    vdir = os.path.join(root, 'data', 'hit_uav_raw')
    results = {}
    for vf in sorted(os.listdir(vdir)):
        if not vf.endswith('.mov'):
            continue
        frames = load_video_gray(os.path.join(vdir, vf), max_frames=120)
        if len(frames) < 9:
            continue
        for k in (4, 8):
            for m in ('linear', 'cubic', 'flow'):
                t0 = time.time()
                p, s = eval_interpolation(frames, k, m)
                results[f'{vf}|k={k}|{m}'] = dict(psnr=round(p, 3), ssim=round(s, 5),
                                                   secs=round(time.time() - t0, 1))
                print(f'{vf:15s} k={k} {m:7s}: PSNR {p:6.2f}  SSIM {s:.4f}  ({time.time()-t0:.0f}s)', flush=True)
    out_p = os.path.join(root, 'experiments', 'e1_interp_results.json')
    json.dump(results, open(out_p, 'w'), indent=1)
    print('saved', out_p)
