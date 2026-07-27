"""E2: aggregate event statistics from converted HIT-UAV + FLIR streams."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from thermal_events.pipeline.convert import load_events
from thermal_events.eval.fidelity import event_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summarize(ed, videos):
    rows = []
    for v in videos:
        for tag in ('raw', 'agc'):
            p = os.path.join(ed, f'{v}_{tag}.npz')
            if not os.path.exists(p):
                continue
            ev, meta = load_events(p)
            st = event_stats(ev, meta.get('height', 512), meta.get('width', 640))
            rows.append(dict(video=v, variant=tag, **st))
            print(f'{v:15s} [{tag}]: {st["n_events"]:9d} ev, rate {st["rate_mean"]:9.0f}/s, '
                  f'pol {st["polarity_pos_frac"]:.3f}, sparsity {st["spatial_sparsity_mean"]:.4f}, '
                  f'burst {st["burstiness"]:.3f}', flush=True)
    return rows


if __name__ == '__main__':
    rows = []
    rows += summarize(os.path.join(ROOT, 'data', 'hit_uav_events'),
                      ['60m-30_1', '120m-30_3', '130m-60_3', '60m-40_3', '70m-90_1'])
    json.dump(rows, open(os.path.join(ROOT, 'experiments', 'e2_hit_uav_stats.json'), 'w'),
              indent=1, default=float)
    print('saved e2_hit_uav_stats.json')
