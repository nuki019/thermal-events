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
                   agc: bool = True, agc_smooth: bool = True,
                   chunk_s: float = 0.0):
    """Full pipeline on a [T,H,W] uint8 sequence. Returns (events dict, meta).

    If chunk_s > 0, the sequence is processed in chunks of chunk_s seconds
    (one-frame overlap to keep the simulator state continuous) to bound peak
    host memory for long videos.
    """
    if chunk_s and chunk_s > 0 and frames_u8.shape[0] > int(chunk_s * fps_in) * 2:
        n_chunk = int(chunk_s * fps_in)
        all_ev = []
        metas = []
        sim = None
        T = frames_u8.shape[0]
        pos = 0
        while pos < T:
            end = min(pos + n_chunk, T)
            seg = frames_u8[pos:end]
            ev, meta = _convert_segment(seg, fps_in, sim_cfg, interp_k,
                                        interp_method, agc, agc_smooth,
                                        sim=sim, keep_sim=True)
            sim = meta.pop('_sim')
            all_ev.append(ev)
            metas.append(meta)
            pos = end if end == T else end - 1
        # shift timestamps by chunk starts
        t_shift = 0.0
        shifted = []
        fps_sim = fps_in * interp_k
        for ci, ev in enumerate(all_ev):
            ev['t'] = ev['t'] + t_shift
            shifted.append(ev)
            # advance by (n_chunk-1)/fps_in except last chunk
            seg_len = (n_chunk - 1) / fps_in if ci < len(all_ev) - 1 else 0
            t_shift += seg_len
        events = {k: np.concatenate([e[k] for e in shifted]) for k in ('t', 'x', 'y', 'p')}
        order = np.argsort(events['t'], kind='stable')
        events = {k: v[order] for k, v in events.items()}
        meta = metas[-1]
        meta['n_events'] = int(len(events['t']))
        meta['chunked'] = True
        meta['frames_in'] = int(frames_u8.shape[0])
        meta['duration_s'] = float(frames_u8.shape[0] / fps_in)
        return events, meta
    events, meta = _convert_segment(frames_u8, fps_in, sim_cfg, interp_k,
                                    interp_method, agc, agc_smooth)
    return events, meta


def _convert_segment(frames_u8, fps_in, sim_cfg, interp_k, interp_method,
                     agc, agc_smooth, sim=None, keep_sim=False):
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
    if sim is None:
        sim = EventSim(sim_cfg, H, W)
    events = sim.run(f, fps_sim)
    meta['fps_sim'] = fps_sim
    meta['n_events'] = int(len(events['t']))
    meta['duration_s'] = float(frames_u8.shape[0] / fps_in)
    meta['height'], meta['width'] = int(H), int(W)
    if keep_sim:
        meta['_sim'] = sim
    return events, meta


def save_events(path: str, events: dict, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, t=events['t'], x=events['x'], y=events['y'],
                        p=events['p'], meta=json.dumps(meta))


def load_events(path: str):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    return dict(t=z['t'], x=z['x'], y=z['y'], p=z['p']), meta
