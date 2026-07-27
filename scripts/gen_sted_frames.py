"""Generate STED frames + boxes with venv (memory-light); events added later."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dataclasses import asdict
from thermal_events.sim.thermal_scene import ThermalScene, SceneConfig
import numpy as np
from thermal_events.sim.thermal_scene import SceneConfig


def sample_scene_cfg(rng, seed, profile='mixed'):
    night = rng.random() < 0.5
    cfg = SceneConfig(
        width=640, height=480, fps=30.0, duration_s=6.0, seed=seed,
        bg_mean=12.0 if night else 25.0,
        bg_tex_amp=float(rng.uniform(1.5, 4.0)),
        bg_drift_rate=float(rng.uniform(0.0, 0.05)),
        n_objects=(int(rng.integers(2, 5)), int(rng.integers(5, 10))),
        obj_temp_contrast=(float(rng.uniform(1.0, 4.0)), float(rng.uniform(6.0, 18.0))),
        obj_frac_hot=0.85 if night else 0.6,
        obj_size_px=(int(rng.integers(10, 20)), int(rng.integers(50, 90))),
        obj_speed_px_s=(float(rng.uniform(0, 20)), float(rng.uniform(60, 200))),
        tau_ms=float(rng.uniform(6, 14)),
        netd_mk=float(rng.uniform(30, 80)),
        prnu=float(rng.uniform(0.005, 0.02)),
    )
    if profile == 'agc':
        cfg.agc_drift_std = float(rng.uniform(0.05, 0.2))
        cfg.agc_jump_prob_per_s = float(rng.uniform(0.5, 2.0))
    else:
        cfg.agc_drift_std = float(rng.uniform(0.0, 0.05))
        cfg.agc_jump_prob_per_s = float(rng.uniform(0.0, 0.3))
    return cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'sted_frames')

def main():
    rng = np.random.default_rng(12345)
    os.makedirs(OUT, exist_ok=True)
    index = []
    if os.path.exists(os.path.join(OUT, 'index.json')):
        index = json.load(open(os.path.join(OUT, 'index.json')))
    done = {e['seq'] for e in index}
    cfgs = []
    for split, n in (('train', 100), ('val', 24)):
        for i in range(n):
            seed = 1000 + (0 if split == 'train' else 50000) + i
            profile = 'agc' if (i % 4 == 3) else 'mixed'
            cfg = sample_scene_cfg(rng, seed, profile)
            sid = f'{split}_{i:04d}'
            cfgs.append((split, i, seed, profile, cfg, sid))
    for split, i, seed, profile, cfg, sid in cfgs:
        if sid in done:
            continue
        import gc; gc.collect()
        out = ThermalScene(cfg).run()
        sdir = os.path.join(OUT, sid)
        os.makedirs(sdir, exist_ok=True)
        np.save(os.path.join(sdir, 'frames.npy'), out['disp8'])
        json.dump(out['boxes'], open(os.path.join(sdir, 'boxes.json'), 'w'))
        smeta = dict(seq=sid, split=split, profile=profile,
                     scene={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                     night=bool(cfg.bg_mean < 18))
        json.dump(smeta, open(os.path.join(sdir, 'meta.json'), 'w'), indent=1)
        index.append(smeta)
        json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
        if len(index) % 10 == 0:
            print(f'{len(index)} seqs', flush=True)
    print('FRAMES_DONE', len(index))

if __name__ == '__main__':
    main()
