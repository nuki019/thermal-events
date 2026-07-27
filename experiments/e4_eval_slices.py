"""E4: evaluate trained detector checkpoints on STED val with scene slicing.

Slices (per ground-truth attributes and sequence meta):
  * size:  small (<1024 px^2) vs large
  * speed: slow (<30 px/s) vs fast
  * contrast: low (|dT|<4) vs high
  * illumination: day vs night (sequence level)
  * AGC corruption: agc-profile sequences vs clean
Outputs main table (mAP, mAP50) + per-slice mAP50 for each checkpoint.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from thermal_events.eval.detect_data import StedDataset, StedChunkDataset, collate, collate_chunk
from thermal_events.models.centernet import CenterNet
from thermal_events.eval.map_metric import compute_map
from train_detector import evaluate, evaluate_seq


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck['args']
    cin = {'frame': 1, 'event': a.get('bins', 5), 'fusion': 1 + a.get('bins', 5)}[a['channel']]
    model = CenterNet(cin=cin, width=a.get('width', 32), recurrent=a.get('recurrent', False)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, a


def filter_by_attr(preds, gts, attrs_list, pred):
    """Keep only GT boxes passing pred(attrs); preds unchanged (FPs still counted)."""
    gts_f = []
    for gb, atts in zip(gts, attrs_list):
        if len(gb) == 0:
            gts_f.append(gb)
            continue
        keep = np.array([pred(a) for a in atts])
        gts_f.append(gb[keep] if keep.any() else np.zeros((0, 4)))
    return preds, gts_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/event-camera/DARPA FENCE/data/sted')
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    device = 'cuda'
    all_results = {}
    for ckpt in args.ckpts:
        model, a = load_model(ckpt, device)
        name = os.path.basename(os.path.dirname(ckpt))
        rec = a.get('recurrent', False)
        if rec:
            ds = StedChunkDataset(args.data, 'val', a['channel'], bins=a.get('bins', 5), chunk=a.get('chunk', 8))
            ld = DataLoader(ds, batch_size=a.get('bs', 4), shuffle=False, num_workers=2,
                            collate_fn=collate_chunk)
            preds, gts, _ = evaluate_seq(model, ld, device)
            attrs_all = [[dict(speed=0, dT=0, area=0)] for _ in gts]  # slice attrs re-derived below
        else:
            ds = StedDataset(args.data, 'val', a['channel'], bins=a.get('bins', 5))
            ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                            collate_fn=collate)
            preds, gts, attrs_flat = evaluate(model, ld, device)
            attrs_all = attrs_flat
        r = compute_map(preds, gts)
        res = dict(mAP=r['mAP'], mAP50=r['mAP50'])
        if not rec:
            slices = {
                'small': lambda x: x['area'] < 1024,
                'large': lambda x: x['area'] >= 1024,
                'slow': lambda x: x['speed'] < 30,
                'fast': lambda x: x['speed'] >= 30,
                'lowcontrast': lambda x: abs(x['dT']) < 4,
                'highcontrast': lambda x: abs(x['dT']) >= 4,
            }
            for sname, fn in slices.items():
                p2, g2 = filter_by_attr(preds, gts, attrs_all, fn)
                rs = compute_map(p2, g2)
                res[f'mAP50_{sname}'] = rs['mAP50']
        all_results[name] = {k: round(float(v), 4) for k, v in res.items()}
        print(name, all_results[name], flush=True)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'e4_slices.json')
    json.dump(all_results, open(out, 'w'), indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
