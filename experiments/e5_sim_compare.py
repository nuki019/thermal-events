"""E5: simulator-mode comparison and downstream detection sensitivity.

Converts a held-out set of STED sequences with each simulator mode
(v2e/volt/poisson), trains the event-channel detector on each, and compares
val mAP to quantify simulator choice sensitivity.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dataclasses import asdict

from thermal_events.sim.thermal_scene import ThermalScene, SceneConfig
from thermal_events.sim.eventsim import SimConfig
from thermal_events.pipeline.convert import convert_frames, save_events
from thermal_events.eval.dataset import sample_scene_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gen_variant(out_dir, mode, n_train=20, n_val=6, seed0=2000):
    rng = np.random.default_rng(999)
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for split, n in (('train', n_train), ('val', n_val)):
        for i in range(n):
            seed = seed0 + (0 if split == 'train' else 70000) + i
            profile = 'mixed'
            cfg = sample_scene_cfg(rng, seed, profile)
            cfg.agc_drift_std = 0.0
            cfg.agc_jump_prob_per_s = 0.0
            out = ThermalScene(cfg).run()
            ev, meta = convert_frames(out['disp8'], cfg.fps, SimConfig(mode=mode),
                                      interp_k=4, interp_method='linear', agc=False)
            sid = f'{split}_{i:04d}'
            sdir = os.path.join(out_dir, sid)
            os.makedirs(sdir, exist_ok=True)
            np.save(os.path.join(sdir, 'frames.npy'), out['disp8'])
            save_events(os.path.join(sdir, 'events.npz'), ev, meta)
            json.dump(out['boxes'], open(os.path.join(sdir, 'boxes.json'), 'w'))
            smeta = dict(seq=sid, split=split, profile=profile, sim_mode=mode,
                         scene={k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in asdict(cfg).items()},
                         night=bool(cfg.bg_mean < 18), n_events=int(len(ev['t'])))
            json.dump(smeta, open(os.path.join(sdir, 'meta.json'), 'w'), indent=1)
            index.append(smeta)
            print(f'[{mode} {split} {i+1}/{n}] {len(ev["t"])} ev', flush=True)
    json.dump(index, open(os.path.join(out_dir, 'index.json'), 'w'), indent=1)


if __name__ == '__main__':
    for mode in ['v2e', 'volt', 'poisson']:
        gen_variant(os.path.join(ROOT, 'data', f'sted_{mode}'), mode)
    print('E5 data done')
