"""AGC de-flicker for thermal video (contribution C1 module).

Thermal cameras apply automatic gain control (AGC) that drifts frame-to-frame
and occasionally re-ranges abruptly, producing *global* photometric flicker.
Fed into a threshold-crossing event simulator, this spawns spurious
scene-wide events that drown genuine motion signal ("Thermal is Always
Wild", CVPR 2026, documents this drift in real thermal video).

Method: pivot-centered single-parameter gain estimation. Thermal AGC pins a
photometric level (typically the histogram middle) and rescales contrast
around it. We fit only the scalar gain per frame from pivot-relative
deviations of a dense percentile profile (21 levels), invert, and keep the
signal in its natural display range (no [0,1] re-stretch, which would flood
the event simulator's log mapping). Correction is gated off when no
significant gain variation is detected. On synthetic scenes with known
ground truth this recovers the radiometric event-count floor exactly.
"""
from __future__ import annotations
import numpy as np


def _robust_stats(frames: np.ndarray, lo_p=2.0, hi_p=98.0):
    """Per-frame median and percentile spread. frames [T,H,W] float."""
    T = frames.shape[0]
    flat = frames.reshape(T, -1)
    med = np.median(flat, axis=1)
    lo = np.percentile(flat, lo_p, axis=1)
    hi = np.percentile(flat, hi_p, axis=1)
    spread = np.maximum(hi - lo, 1e-3)
    return med.astype(np.float64), spread.astype(np.float64)


def _medfilt1d(x: np.ndarray, win: int) -> np.ndarray:
    """Pure-numpy median filter (odd win), edge-padded."""
    win = int(win) | 1
    pad = win // 2
    xp = np.pad(x, pad, mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, win), axis=1)


def _jump_robust_smooth(x: np.ndarray, win=9, alpha=0.2, jump_sigma=4.0):
    """Median filter + exponential smoothing; resets smoothing state at jumps."""
    xm = _medfilt1d(x, win)
    out = np.empty_like(xm)
    out[0] = xm[0]
    resid = xm - np.concatenate([[xm[0]], xm[:-1]])
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    for i in range(1, len(xm)):
        if abs(resid[i]) > jump_sigma * 1.4826 * mad:
            out[i] = xm[i]          # change point: accept immediately
        else:
            out[i] = out[i - 1] + alpha * (xm[i] - out[i - 1])
    return out


def agc_normalize(frames_u8: np.ndarray, win=5, alpha=0.35, smooth=False,
                  gate_sigma: float = 0.01):
    """Normalize a [T,H,W] uint8 sequence to temporally stable photometry.

    AGC model (per frame):  disp = g * (q - pivot) + pivot,
    where pivot is the photometric pinning level (histogram middle for typical
    thermal AGC). We estimate pivot as the temporal median of per-frame
    medians, fit the scalar gain g per frame from pivot-relative deviations
    of a dense percentile profile (21 levels, least squares), optionally
    smooth the gain trace (jump-robust), and invert:
        q = pivot + (disp - pivot) / g.

    Single-parameter pivot-centered correction keeps bulk (near-pivot) pixels
    pinned even under gain-estimation noise, avoiding correction-induced
    event floods. If the estimated gain trace is nearly constant
    (|g-1| < gate_sigma throughout), correction is skipped (flicker gate).

    Returns (norm_frames float32 in [0,1], diag dict with gain trace/pivot).
    """
    f = frames_u8.astype(np.float32)
    T = f.shape[0]
    # percentile profile via subsampled pixels (bounded memory on long videos)
    max_pix = 40000
    flat = f.reshape(T, -1)
    if flat.shape[1] > max_pix:
        cols = np.linspace(0, flat.shape[1] - 1, max_pix).astype(np.int64)
        flat = flat[:, cols]
    qs = list(np.linspace(2, 98, 21))
    P = np.percentile(flat, qs, axis=1)                        # [21,T]
    ref = np.median(P, axis=1)                               # [21]
    piv = float(np.median(P[len(qs) // 2]))
    a = ref - piv                                            # [21]
    b = P - piv                                              # [21,T]
    gain = (a[:, None] * b).sum(axis=0) / max((a * a).sum(), 1e-6)
    g_med = np.median(gain)
    gain = np.clip(gain, 0.2 * g_med, 5.0 * g_med)
    if smooth:
        gain = _jump_robust_smooth(gain, win=win, alpha=alpha, jump_sigma=5.0)
    if np.all(np.abs(gain - 1.0) < gate_sigma):
        norm = f
        gain = np.ones_like(gain)
    else:
        norm = f
        g32 = np.maximum(gain, 1e-3).astype(np.float32)
        np.subtract(norm, np.float32(piv), out=norm)
        norm /= g32[:, None, None]
        norm += np.float32(piv)
    # Keep the corrected sequence in the *natural* mid display range of the
    # reference profile. Do NOT re-stretch to [0,1]: stretching pushes the
    # scene toward zero where the event simulator's log mapping is
    # hyper-sensitive (dlogI = dI/I), flooding the stream with noise events.
    np.clip(norm, 2.0, 253.0, out=norm)
    norm /= np.float32(255.0)
    return norm, dict(gain=gain, pivot=piv, ref=ref, percentiles=P)


def flicker_index(frames: np.ndarray) -> float:
    """Mean absolute frame-to-frame change of global percentiles (flicker proxy).

    Averaged over the 10/25/50/75/90th percentiles so both offset- and
    gain-type flicker are captured.
    """
    f = frames.reshape(frames.shape[0], -1)
    p = np.percentile(f, [10, 25, 50, 75, 90], axis=1)  # [5,T]
    return float(np.mean(np.abs(np.diff(p, axis=1))))
