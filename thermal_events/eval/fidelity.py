"""Fidelity metrics for synthetic thermal event streams (C3).

Without real thermal event hardware data, fidelity is assessed by:
  * event-rate statistics (mean/median/peak rate, burstiness)
  * polarity balance
  * spatial sparsity (fraction of active pixels per time bin)
  * inter-event interval distribution (log-binned histogram)
  * spatial event-density correlation with temporal image gradients
    (events should live on moving thermal edges): the Edge-Event Alignment
    (EEA) score we define as the Pearson correlation between the event-count
    map and the spatiotemporal gradient magnitude map of the source video.
  * comparison against statistical envelopes of real visible-light DVS data
    (Gen1/1Mpx literature values) as a sanity envelope.
"""
from __future__ import annotations
import numpy as np


def event_stats(events: dict, h: int, w: int, bin_s: float = 0.05) -> dict:
    t, x, y, p = events['t'], events['x'], events['y'], events['p']
    if len(t) == 0:
        return dict(n_events=0)
    dur = t.max() - t.min() + 1e-9
    nb = max(1, int(dur / bin_s))
    bins = np.floor((t - t.min()) / bin_s).astype(int)
    counts = np.bincount(bins, minlength=nb).astype(np.float64)
    # spatial activity per bin (sample up to 200 bins)
    sparsity = []
    step = max(1, nb // 200)
    for b in range(0, nb, step):
        m = bins == b
        if m.sum():
            sparsity.append(len(np.unique(y[m] * w + x[m])) / (h * w))
    iei = np.diff(t)
    pos_frac = float((p > 0).mean())
    return dict(
        n_events=int(len(t)),
        duration_s=float(dur),
        rate_mean=float(counts.mean() / bin_s),
        rate_median=float(np.median(counts) / bin_s),
        rate_p99=float(np.percentile(counts, 99) / bin_s),
        burstiness=float(counts.std() / max(counts.mean(), 1e-9)),
        polarity_pos_frac=pos_frac,
        spatial_sparsity_mean=float(np.mean(sparsity)) if sparsity else 0.0,
        iei_median_us=float(np.median(iei) * 1e6) if len(iei) else 0.0,
        iei_p01_us=float(np.percentile(iei, 1) * 1e6) if len(iei) else 0.0,
        mpix_per_s=float(len(t) / dur / (h * w)),
    )


def edge_event_alignment(events: dict, frames_u8: np.ndarray, fps: float) -> float:
    """Pearson corr between event-count map and spatiotemporal gradient map."""
    import cv2
    t, x, y = events['t'], events['x'], events['y']
    T, H, W = frames_u8.shape
    if len(t) == 0:
        return 0.0
    ev_map = np.zeros((H, W), np.float64)
    np.add.at(ev_map, (y, x), 1.0)
    f = frames_u8.astype(np.float32)
    grad = np.zeros((H, W), np.float32)
    for i in range(T):
        gx = cv2.Sobel(f[i], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f[i], cv2.CV_32F, 0, 1, ksize=3)
        if i < T - 1:
            m = (np.abs(f[i + 1] - f[i]) > 0.5).astype(np.float32)
        else:
            m = np.ones((H, W), np.float32)
        grad += np.sqrt(gx ** 2 + gy ** 2) * m
    grad /= max(T, 1)
    a = ev_map.ravel() - ev_map.mean()
    b = grad.ravel() - grad.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0 else 0.0
