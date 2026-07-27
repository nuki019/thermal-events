"""E3b: threshold<->NETD calibration curve (object-contrast event yield)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from thermal_events.sim.thermal_scene import ThermalScene, SceneConfig
from thermal_events.sim.eventsim import SimConfig
from thermal_events.pipeline.convert import convert_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    out_dir = os.path.join(ROOT, 'experiments', 'results', 'e3')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    dTs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    for th in [0.1, 0.2, 0.3]:
        for dT in dTs:
            cfg = SceneConfig(duration_s=3.0, fps=30.0, seed=777, n_objects=(3, 3),
                              obj_frac_hot=1.0,
                              obj_temp_contrast=(dT, dT + 1e-6), obj_speed_px_s=(20, 40),
                              netd_mk=30.0, agc_drift_std=0.0, agc_jump_prob_per_s=0.0,
                              bg_drift_rate=0.0)
            out = ThermalScene(cfg).run()
            frames = out['disp8']
            ev, _ = convert_frames(frames, 30.0, SimConfig(mode='v2e', th_on=th, th_off=th, shot_noise=0.0),
                                   interp_k=4, interp_method='linear', agc=False)
            H, W = frames.shape[1:]
            mask = np.zeros((H, W), bool)
            for bl in out['boxes']:
                for b in bl:
                    x0 = max(0, int(b['cx'] - b['w'])); x1 = min(W, int(b['cx'] + b['w']))
                    y0 = max(0, int(b['cy'] - b['h'])); y1 = min(H, int(b['cy'] + b['h']))
                    mask[y0:y1, x0:x1] = True
            in_obj = mask[ev['y'], ev['x']] if len(ev['t']) else np.zeros(0, bool)
            yield_frac = float(in_obj.mean()) if len(ev['t']) else 0.0
            rows.append(dict(th=th, dT=dT, n_events=len(ev['t']),
                             rate=len(ev['t']) / cfg.duration_s, in_obj_frac=yield_frac))
            print(f'th={th} dT={dT}: {len(ev["t"])} ev, in-obj {yield_frac:.3f}', flush=True)
    json.dump(rows, open(os.path.join(out_dir, 'e3_calibration.json'), 'w'), indent=1)
    print('saved e3_calibration.json')

if __name__ == '__main__':
    main()
