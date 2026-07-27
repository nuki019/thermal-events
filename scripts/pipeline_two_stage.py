"""Two-stage memory-safe converter: stage A (venv, CPU) AGC+interp -> npy;
stage B (anaconda, GPU) event simulation from npy. For HIT-UAV + FLIR."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def stage_a(frames_path, out_path, k, agc=True, smooth=True):
    """CPU-only: load frames, AGC normalize, linear-interpolate, save uint8."""
    from thermal_events.pipeline.agc import agc_normalize
    from thermal_events.pipeline.interp import interpolate
    frames = np.load(frames_path, mmap_mode='r')
    f = np.array(frames)  # load fully (uint8)
    if agc:
        norm, _ = agc_normalize(f, smooth=smooth)
        f = (norm * 255).astype(np.uint8)
        del norm
    if k > 1:
        f = interpolate(f, k, 'linear')
    np.save(out_path, f)
    print('stage A ->', out_path, f.shape, flush=True)


def stage_b(frames_path, out_path, fps_sim, mode='v2e', th=0.2, shot=0.01, lp=0.0):
    """GPU: event simulation from interpolated frames."""
    from thermal_events.sim.eventsim import EventSim, SimConfig
    from thermal_events.pipeline.convert import save_events
    f = np.load(frames_path, mmap_mode='r')
    H, W = f.shape[1:]
    sim = EventSim(SimConfig(mode=mode, th_on=th, th_off=th, shot_noise=shot,
                             lp_cutoff_hz=lp), H, W)
    # stream blocks to bound GPU tensor memory; copy via pinned staging buffer
    import torch
    ts, xs, ys, ps = [], [], [], []
    block = 60
    staging = np.empty((block, H, W), np.uint8)
    for s0 in range(0, f.shape[0], block):
        n = min(block, f.shape[0] - s0)
        staging[:n] = f[s0:s0 + n]
        seg = staging[:n]
        ev = sim.run(seg, fps_sim)
        # shift timestamps
        ev['t'] = ev['t'] + s0 / fps_sim
        ts.append(ev['t']); xs.append(ev['x']); ys.append(ev['y']); ps.append(ev['p'])
    t = np.concatenate(ts); x = np.concatenate(xs); y = np.concatenate(ys); p = np.concatenate(ps)
    order = np.argsort(t, kind='stable')
    meta = dict(fps_sim=fps_sim, n_events=int(len(t)), mode=mode, th=th, shot=shot, lp=lp)
    save_events(out_path, dict(t=t[order], x=x[order], y=y[order], p=p[order]), meta)
    print('stage B ->', out_path, len(t), 'events', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['a', 'b'])
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--fps', type=float, default=60.0)
    ap.add_argument('--mode', default='v2e')
    ap.add_argument('--th', type=float, default=0.2)
    ap.add_argument('--shot', type=float, default=0.01)
    ap.add_argument('--lp', type=float, default=0.0)
    ap.add_argument('--no-agc', action='store_true')
    args = ap.parse_args()
    if args.stage == 'a':
        stage_a(args.frames, args.out, args.k, agc=not args.no_agc)
    else:
        stage_b(args.frames, args.out, args.fps, args.mode, args.th, args.shot, args.lp)
