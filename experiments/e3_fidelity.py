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
