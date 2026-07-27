"""STED: Synthetic Thermal Event detection benchmark generator.

Generates N sequences with diverse conditions; stores per-sequence:
  * disp8 frames (uint8 [T,H,W])         - frame channel
  * event stream (npz t,x,y,p)           - event channel (v2e default)
  * boxes per frame (json)               - ground truth
  * seq meta (day/night, AGC level etc.) - for slice analysis
"""
from __future__ import annotations
import os
import json
import numpy as np
from dataclasses import asdict

from ..sim.thermal_scene import ThermalScene, SceneConfig
from ..sim.eventsim import SimConfig
from ..pipeline.convert import convert_frames, save_events


def sample_scene_cfg(rng: np.random.Generator, seed: int, profile: str = 'mixed') -> SceneConfig:
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


def generate_benchmark(out_dir: str, n_train=100, n_val=24, sim_cfg: SimConfig | None = None,
                       interp_k=4, interp_method='linear', seed0=1000):
    if sim_cfg is None:
        sim_cfg = SimConfig(mode='v2e')
    rng = np.random.default_rng(12345)
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for split, n in (('train', n_train), ('val', n_val)):
        for i in range(n):
            seed = seed0 + (0 if split == 'train' else 50000) + i
            profile = 'agc' if (i % 4 == 3) else 'mixed'
            cfg = sample_scene_cfg(rng, seed, profile)
            scene = ThermalScene(cfg)
            out = scene.run()
            events, meta = convert_frames(out['disp8'], cfg.fps, sim_cfg,
                                          interp_k=interp_k, interp_method=interp_method)
            sid = f'{split}_{i:04d}'
            sdir = os.path.join(out_dir, sid)
            os.makedirs(sdir, exist_ok=True)
            np.save(os.path.join(sdir, 'frames.npy'), out['disp8'])
            save_events(os.path.join(sdir, 'events.npz'), events, meta)
            json.dump(out['boxes'], open(os.path.join(sdir, 'boxes.json'), 'w'))
            smeta = dict(seq=sid, split=split, profile=profile,
                         scene={k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in asdict(cfg).items()},
                         night=bool(cfg.bg_mean < 18),
                         n_events=int(len(events['t'])))
            json.dump(smeta, open(os.path.join(sdir, 'meta.json'), 'w'), indent=1)
            index.append(smeta)
            print(f'[{split} {i+1}/{n}] {sid}: {len(events["t"])} ev, '
                  f'{sum(len(b) for b in out["boxes"])} boxes', flush=True)
    json.dump(index, open(os.path.join(out_dir, 'index.json'), 'w'), indent=1)
    return index
