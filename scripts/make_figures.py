"""Generate paper figures (PDF, 300dpi-friendly) from data and results."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 200, 'savefig.bbox': 'tight'})


def fig_hit_uav_events():
    """Panel: HIT-UAV thermal frame + raw events + AGC-normalized events."""
    from thermal_events.pipeline.convert import load_events
    vdir = os.path.join(ROOT, 'data', 'hit_uav_raw')
    ed = os.path.join(ROOT, 'data', 'hit_uav_events')
    vid = '60m-30_1'
    cap = cv2.VideoCapture(os.path.join(vdir, vid + '.mov'))
    frames = []
    for _ in range(40):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    frame = frames[30]
    H, W = frame.shape
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    axes[0].imshow(frame, cmap='inferno', vmin=0, vmax=255)
    axes[0].set_title('(a) thermal frame')
    for ax, tag, ttl in ((axes[1], 'raw', '(b) events, raw'),
                         (axes[2], 'agc', '(c) events, AGC de-flickered')):
        p = os.path.join(ed, f'{vid}_{tag}.npz')
        if not os.path.exists(p):
            ax.set_title(ttl + ' [missing]')
            continue
        ev, meta = load_events(p)
        t1 = 30 / 7.5
        m = (ev['t'] >= t1 - 0.133) & (ev['t'] < t1)
        img = np.full((H, W, 3), 0.5, np.float32)
        yy, xx, pp = ev['y'][m], ev['x'][m], ev['p'][m]
        img[yy[pp > 0], xx[pp > 0]] = [1, 0.3, 0.3]
        img[yy[pp < 0], xx[pp < 0]] = [0.3, 0.5, 1]
        ax.imshow(img)
        ax.set_title(f'{ttl}\n{m.sum()} ev/133ms')
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(FIG, 'hit_uav_events.pdf'))
    plt.close(fig)
    print('fig_hit_uav_events done')


def fig_agc_gain_trace():
    """AGC gain trace on a HIT-UAV video + event-rate time series raw vs agc."""
    from thermal_events.pipeline.convert import load_events
    from thermal_events.pipeline.agc import agc_normalize
    vdir = os.path.join(ROOT, 'data', 'hit_uav_raw')
    vid = '60m-30_1'
    cap = cv2.VideoCapture(os.path.join(vdir, vid + '.mov'))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    frames = np.stack(frames)
    _, diag = agc_normalize(frames, smooth=True)
    ed = os.path.join(ROOT, 'data', 'hit_uav_events')
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 3.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.4]))
    tt = np.arange(len(diag['gain'])) / 7.5
    axes[0].plot(tt, diag['gain'], lw=1)
    axes[0].set_ylabel('est. AGC gain')
    for tag, c, lbl in (('raw', '0.6', 'raw'), ('agc', 'tab:red', 'AGC-normalized')):
        p = os.path.join(ed, f'{vid}_{tag}.npz')
        if not os.path.exists(p):
            continue
        ev, _ = load_events(p)
        t0, tmax = ev['t'].min(), ev['t'].max()
        bins = np.arange(t0, tmax, 0.5)
        h, _ = np.histogram(ev['t'], bins)
        axes[1].plot(bins[:-1], h / 0.5 / 1e3, lw=0.8, color=c, label=lbl)
    axes[1].set_ylabel('event rate [kev/s]')
    axes[1].set_xlabel('time [s]')
    axes[1].legend(frameon=False)
    fig.savefig(os.path.join(FIG, 'agc_gain_trace.pdf'))
    plt.close(fig)
    print('fig_agc_gain_trace done')


def fig_e1_interp():
    d = json.load(open(os.path.join(ROOT, 'experiments', 'e1_interp_results.json')))
    vids = sorted({k.split('|')[0] for k in d})
    methods = ['linear', 'cubic', 'flow']
    x = np.arange(len(vids))
    w = 0.25
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    for i, m in enumerate(methods):
        vals = [d[f'{v}|k=4|{m}']['psnr'] for v in vids]
        ax.bar(x + (i - 1) * w, vals, w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('.mov', '').replace('_', ' ') for v in vids], rotation=20, ha='right')
    ax.set_ylabel('held-out PSNR [dB]')
    ax.legend(frameon=False, ncol=3)
    ax.set_title('Interpolation on real thermal video (7.5 Hz, 4x)')
    fig.savefig(os.path.join(FIG, 'e1_interp.pdf'))
    plt.close(fig)
    print('fig_e1_interp done')


if __name__ == '__main__':
    fig_hit_uav_events()
    fig_agc_gain_trace()
    fig_e1_interp()
