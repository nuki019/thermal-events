"""Convert HIT-UAV sample thermal videos to event streams (E2, real anchor).

Runs AGC on/off variants for the de-flicker study and computes event stats.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2

from thermal_events.sim.eventsim import SimConfig
from thermal_events.pipeline.convert import convert_frames, save_events
from thermal_events.pipeline.agc import agc_normalize, flicker_index
from thermal_events.eval.fidelity import event_stats, edge_event_alignment


def load_video_gray(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    return np.stack(out), cap.get(cv2.CAP_PROP_FPS)


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vdir = os.path.join(root, 'data', 'hit_uav_raw')
    odir = os.path.join(root, 'data', 'hit_uav_events')
    os.makedirs(odir, exist_ok=True)
    rows = []
    for vf in sorted(os.listdir(vdir)):
        if not vf.endswith('.mov'):
            continue
        frames, fps = load_video_gray(os.path.join(vdir, vf))
        fps = 7.5
        H, W = frames.shape[1:]
        fl_before = flicker_index(frames[::4].astype(np.float32))
        norm, diag = agc_normalize(frames, smooth=True)
        fl_after = flicker_index((norm[::4] * 255).astype(np.float32))
        del norm
        for agc_on in (False, True):
            sc = SimConfig(mode='v2e')
            t0 = time.time()
            ev, meta = convert_frames(frames, fps, sc, interp_k=8, interp_method='linear',
                                      agc=agc_on, agc_smooth=True, chunk_s=10.0)
            st = event_stats(ev, H, W)
            eea = edge_event_alignment(ev, frames, fps)
            tag = 'agc' if agc_on else 'raw'
            save_events(os.path.join(odir, f'{vf[:-4]}_{tag}.npz'), ev, meta)
            rows.append(dict(video=vf, variant=tag, flicker_before=fl_before,
                             flicker_after=fl_after, secs=round(time.time() - t0, 1),
                             eea=eea, **{k: v for k, v in st.items()}))
            print(f'{vf} [{tag}]: {st["n_events"]} ev, rate {st["rate_mean"]:.0f}/s, '
                  f'pol {st["polarity_pos_frac"]:.3f}, EEA {eea:.3f}, '
                  f'flicker {fl_before:.3f}->{fl_after:.3f}', flush=True)
    json.dump(rows, open(os.path.join(odir, 'hit_uav_event_stats.json'), 'w'), indent=1, default=float)
    print('saved stats')
