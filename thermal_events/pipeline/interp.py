"""Temporal frame interpolation for event simulation (E1 module).

Event simulators need high temporal sampling; source thermal video is often
7.5-30 Hz. We compare:
  * 'linear' : per-pixel linear blending (baseline)
  * 'cubic'  : per-pixel cubic (Catmull-Rom) temporal interpolation
  * 'flow'   : optical-flow-guided bidirectional warping (Farneback flow,
               occlusion-aware blending) - a classical stand-in for learned
               interpolators (SuperSloMo/RIFE), whose visible-light-trained
               weights are of questionable transfer to thermal (an E1 finding).

API: interpolate(frames [T,H,W] uint8, k) -> [k*(T-1)+1, H, W] uint8
"""
from __future__ import annotations
import numpy as np
import cv2


def _linear(frames: np.ndarray, k: int) -> np.ndarray:
    T, H, W = frames.shape
    out = np.empty(((T - 1) * k + 1, H, W), np.float32)
    out[0] = frames[0]
    for i in range(T - 1):
        a = frames[i].astype(np.float32)
        b = frames[i + 1].astype(np.float32)
        for j in range(k):
            t = j / k
            out[i * k + j] = (1 - t) * a + t * b
    out[-1] = frames[-1]
    return np.clip(out, 0, 255).astype(np.uint8)


def _cubic(frames: np.ndarray, k: int) -> np.ndarray:
    T, H, W = frames.shape
    f = frames.astype(np.float32)
    out = np.empty(((T - 1) * k + 1, H, W), np.float32)
    for i in range(T - 1):
        f0 = f[max(i - 1, 0)]; f1 = f[i]; f2 = f[i + 1]; f3 = f[min(i + 2, T - 1)]
        # Catmull-Rom coefficients
        for j in range(k):
            t = j / k
            t2, t3 = t * t, t * t * t
            out[i * k + j] = (0.5 * ((2 * f1) + (-f0 + f2) * t
                              + (2 * f0 - 5 * f1 + 4 * f2 - f3) * t2
                              + (-f0 + 3 * f1 - 3 * f2 + f3) * t3))
    out[-1] = f[-1]
    return np.clip(out, 0, 255).astype(np.uint8)


def _warp(img: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    """Warp img by t*flow (pixel displacement)."""
    H, W = img.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx = gx + t * flow[..., 0]
    my = gy + t * flow[..., 1]
    return cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _flow_guided(frames: np.ndarray, k: int) -> np.ndarray:
    T, H, W = frames.shape
    out = np.empty(((T - 1) * k + 1, H, W), np.uint8)
    out[0] = frames[0]
    for i in range(T - 1):
        a = frames[i]
        b = frames[i + 1]
        fa = a.astype(np.float32)
        fb = b.astype(np.float32)
        flow_ab = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_ba = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow_ab, axis=2) + np.linalg.norm(flow_ba, axis=2)
        conf = np.exp(-mag / 20.0).astype(np.float32)          # low motion -> trust flow
        for j in range(1, k):
            t = j / k
            wa = _warp(fa, flow_ab, t)                          # forward from a
            wb = _warp(fb, flow_ba, 1.0 - t)                    # backward from b
            blend = (1 - t) * wa + t * wb
            lin = (1 - t) * fa + t * fb
            out[i * k + j] = np.clip(conf * blend + (1 - conf) * lin, 0, 255).astype(np.uint8)
    out[-1] = frames[-1]
    return out


def interpolate(frames: np.ndarray, k: int, method: str = 'flow') -> np.ndarray:
    """Upsample [T,H,W] uint8 by integer factor k -> [k*(T-1)+1,H,W] uint8."""
    if k == 1:
        return frames.copy()
    if method == 'linear':
        return _linear(frames, k)
    if method == 'cubic':
        return _cubic(frames, k)
    if method == 'flow':
        return _flow_guided(frames, k)
    raise ValueError(method)


def eval_interpolation(frames: np.ndarray, k: int, method: str):
    """Held-out evaluation: drop every k-th interior frame, reconstruct, score.

    Returns (psnr, ssim) on held-out frames.
    """
    from skimage.metrics import structural_similarity as ssim
    T = frames.shape[0]
    keep_idx = np.arange(0, T, k)
    keep = frames[keep_idx]
    if len(keep) < 3:
        return np.nan, np.nan
    # map original index -> position in upsampled sequence
    up = interpolate(keep, k, method)
    psnrs, ssims = [], []
    for m in range(1, k):
        orig_idx = np.arange(m, T - 1 + 1, k)
        orig_idx = orig_idx[orig_idx < T]
        for oi in orig_idx:
            up_idx = (oi // k) * k + (oi % k)
            if up_idx >= len(up):
                continue
            gt = frames[oi].astype(np.float32)
            pr = up[up_idx].astype(np.float32)
            mse = np.mean((gt - pr) ** 2)
            psnrs.append(10 * np.log10(255.0 ** 2 / max(mse, 1e-6)))
            ssims.append(ssim(gt, pr, data_range=255.0))
    return float(np.mean(psnrs)), float(np.mean(ssims))
