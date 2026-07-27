"""ThermEv conversion pipeline driver (C1).

video frames -> [AGC de-flicker] -> [temporal interpolation] -> event
simulation -> compressed event store (.npz with t,x,y,p + metadata).
"""
from __future__ import annotations
import os
import json
import numpy as np
from dataclasses import asdict

from ..sim.eventsim import EventSim, SimConfig
from .agc import agc_normalize
from .interp import interpolate


def convert_frames(frames_u8: np.ndarray, fps_in: float, sim_cfg: SimConfig,
                   interp_k: int = 8, interp_method: str = 'flow',
                   agc: bool = True, agc_smooth: bool = True):
    """Full pipeline on a [T,H,W] uint8 sequence. Returns (events dict, meta)."""
    meta = dict(fps_in=fps_in, interp_k=interp_k, interp_method=interp_method,
                agc=agc, sim=asdict(sim_cfg), frames_in=int(frames_u8.shape[0]))
    f = frames_u8
    if agc:
        norm, diag = agc_normalize(f, smooth=agc_smooth)
        f = (norm * 255.0).astype(np.uint8)
        meta['agc_gain_trace'] = diag['gain'].round(4).tolist()
        meta['agc_pivot'] = round(float(diag['pivot']), 2)
    if interp_k > 1:
        f = interpolate(f, interp_k, interp_method)
    fps_sim = fps_in * interp_k
    H, W = f.shape[1:]
    sim = EventSim(sim_cfg, H, W)
    events = sim.run(f, fps_sim)
    meta['fps_sim'] = fps_sim
    meta['n_events'] = int(len(events['t']))
    meta['duration_s'] = float(frames_u8.shape[0] / fps_in)
    meta['height'], meta['width'] = int(H), int(W)
    return events, meta


def save_events(path: str, events: dict, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, t=events['t'], x=events['x'], y=events['y'],
                        p=events['p'], meta=json.dumps(meta))


def load_events(path: str):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    return dict(t=z['t'], x=z['x'], y=z['y'], p=z['p']), meta
