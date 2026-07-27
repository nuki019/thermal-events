"""E3: fidelity + parameter sensitivity of thermal event simulation.

Grid over simulator mode x contrast threshold x shot noise x bolometer
low-pass cutoff on a fixed set of synthetic sequences (known ground truth).
Outputs per-config event statistics (rate, polarity, sparsity, EEA) and the
NETD->threshold mapping sanity curve.
"""
import sys, os, json, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from thermal_events.sim.thermal_scene import ThermalScene, SceneConfig
from thermal_events.sim.eventsim import SimConfig
from thermal_events.pipeline.convert import convert_frames
from thermal_events.eval.fidelity import event_stats, edge_event_alignment


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, 'experiments', 'results', 'e3')
    os.makedirs(out_dir, exist_ok=True)
    # 3 diverse sequences
    cfgs = [SceneConfig(duration_s=3.0, fps=30.0, seed=101, agc_drift_std=0.0, agc_jump_prob_per_s=0.0),
            SceneConfig(duration_s=3.0, fps=30.0, seed=202, bg_mean=12.0, obj_frac_hot=0.9,
                        agc_drift_std=0.0, agc_jump_prob_per_s=0.0),
            SceneConfig(duration_s=3.0, fps=30.0, seed=303, obj_speed_px_s=(60, 220),
                        agc_drift_std=0.0, agc_jump_prob_per_s=0.0)]
    scenes = []
    for c in cfgs:
        out = ThermalScene(c).run()
        scenes.append(out['disp8'])
    grid = []
    for mode in ['v2e', 'volt', 'poisson']:
        for th in [0.1, 0.2, 0.3]:
            for shot in [0.0, 0.02]:
                for lp in [0.0, 30.0]:     # 0=off, 30Hz ~ tau 5ms-ish
                    grid.append(dict(mode=mode, th_on=th, th_off=th, shot_noise=shot, lp_cutoff_hz=lp))
    rows = []
    for gi, g in enumerate(grid):
        for si, frames in enumerate(scenes):
            sc = SimConfig(mode=g['mode'], th_on=g['th_on'], th_off=g['th_off'],
                           shot_noise=g['shot_noise'], lp_cutoff_hz=g['lp_cutoff_hz'])
            ev, meta = convert_frames(frames, 30.0, sc, interp_k=4, interp_method='linear', agc=False)
            st = event_stats(ev, 480, 640)
            eea = edge_event_alignment(ev, frames, 30.0)
            row = dict(grid=g, scene=si, **{k: v for k, v in st.items()}, eea=eea)
            rows.append(row)
        print(f'[{gi+1}/{len(grid)}] {g} -> rate {st["rate_mean"]:.0f} ev/s, pol {st["polarity_pos_frac"]:.3f}, EEA {eea:.3f}', flush=True)
    json.dump(rows, open(os.path.join(out_dir, 'e3_grid.json'), 'w'), indent=1, default=float)
    print('saved', os.path.join(out_dir, 'e3_grid.json'))


if __name__ == '__main__':
    main()


def calibration_curve():
    """E3b: threshold<->NETD calibration via object-contrast response.

    For each simulator threshold theta, generate scenes whose objects have
    swept temperature contrast dT and measure the event yield within object
    boxes. The dT at which the yield reaches 50% of its plateau defines the
    minimum detectable contrast, giving an empirical theta<->NETD mapping.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, 'experiments', 'results', 'e3')
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    dTs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    for th in [0.1, 0.2, 0.3]:
        for dT in dTs:
            cfg = SceneConfig(duration_s=3.0, fps=30.0, seed=777, n_objects=(3, 3),
                              obj_temp_contrast=(dT, dT + 1e-6), obj_speed_px_s=(20, 40),
                              netd_mk=30.0, agc_drift_std=0.0, agc_jump_prob_per_s=0.0,
                              bg_drift_rate=0.0)
            out = ThermalScene(cfg).run()
            frames = out['disp8']
            sc = SimConfig(mode='v2e', th_on=th, th_off=th, shot_noise=0.0)
            ev, _ = convert_frames(frames, 30.0, sc, interp_k=4, interp_method='linear', agc=False)
            # event yield inside union of object boxes (any frame)
            H, W = frames.shape[1:]
            mask = np.zeros((H, W), bool)
            for bl in out['boxes']:
                for b in bl:
                    x0 = max(0, int(b['cx'] - b['w'])); x1 = min(W, int(b['cx'] + b['w']))
                    y0 = max(0, int(b['cy'] - b['h'])); y1 = min(H, int(b['cy'] + b['h']))
                    mask[y0:y1, x0:x1] = True
            in_obj = mask[ev['y'], ev['x']] if len(ev['t']) else np.zeros(0, bool)
            yield_frac = float(in_obj.mean()) if len(ev['t']) else 0.0
            rate = len(ev['t']) / cfg.duration_s
            rows.append(dict(th=th, dT=dT, n_events=len(ev['t']), rate=rate,
                             in_obj_frac=yield_frac))
            print(f'th={th} dT={dT}: {len(ev["t"])} ev, in-obj {yield_frac:.3f}', flush=True)
    json.dump(rows, open(os.path.join(out_dir, 'e3_calibration.json'), 'w'), indent=1)
    print('saved calibration')
