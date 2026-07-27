"""Train/evaluate CenterNet-lite on STED channels (frame/event/fusion, ±recurrent)."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from thermal_events.eval.detect_data import StedDataset, StedChunkDataset, collate, collate_chunk
from thermal_events.models.centernet import CenterNet, build_targets, focal_loss, reg_l1, decode
from thermal_events.eval.map_metric import compute_map


def run_epoch_seq(model, loader, opt, scaler, device, stride=4):
    """Recurrent training over sequence chunks (state reset per chunk)."""
    tot = {'loss': 0.0, 'hm': 0.0, 'wh': 0.0, 'off': 0.0, 'n': 0}
    model.train()
    for xs_seq, boxes_seq, _ in loader:
        state = None
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        nsteps = len(xs_seq)
        for xs, boxes_list in zip(xs_seq, boxes_seq):
            xs = xs.to(device, non_blocking=True)
            B = xs.shape[0]
            with torch.autocast('cuda'):
                out, state = model(xs, state)
                state = tuple(s.detach() for s in state) if state is not None else None
                hm_t, wh_t, off_t, ind, mask = [], [], [], [], []
                for b in boxes_list:
                    t = build_targets(b.tolist(), xs.shape[2], xs.shape[3], stride, device=device)
                    hm_t.append(t[0]); wh_t.append(t[1]); off_t.append(t[2]); ind.append(t[3]); mask.append(t[4])
                hm_t = torch.stack(hm_t); wh_t = torch.stack(wh_t); off_t = torch.stack(off_t)
                ind = torch.stack(ind); mask = torch.stack(mask)
                l_hm = focal_loss(out['hm'], hm_t)
                l_wh = reg_l1(out['wh'], ind, mask, wh_t)
                l_off = reg_l1(out['off'], ind, mask, off_t)
                loss = l_hm + 0.1 * l_wh + l_off
            scaler.scale(loss / nsteps).backward()
            loss_acc += float(loss)
            tot['loss'] += float(loss) * B; tot['hm'] += float(l_hm) * B
            tot['wh'] += float(l_wh) * B; tot['off'] += float(l_off) * B; tot['n'] += B
        scaler.step(opt)
        scaler.update()
    return {k: v / max(tot['n'], 1) for k, v in tot.items() if k != 'n'}


@torch.no_grad()
def evaluate_seq(model, loader, device, stride=4, thresh=0.2):
    """Recurrent eval: run full sequences in order, carrying state."""
    model.eval()
    preds, gts, attrs_all = [], [], []
    for xs_seq, boxes_seq, _ in loader:
        state = None
        for xs, boxes_list in zip(xs_seq, boxes_seq):
            xs = xs.to(device)
            with torch.autocast('cuda'):
                out, state = model(xs, state)
            dets = decode({k: v.float() for k, v in out.items()}, stride=stride, thresh=thresh)
            for b, bb in enumerate(boxes_list):
                xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2,
                                    bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2], 1).numpy()
                gts.append(xyxy)
                preds.append(np.array(dets[b]) if dets[b] else np.zeros((0, 5)))
    return preds, gts, None


def run_epoch(model, loader, opt, scaler, device, train=True, stride=4):
    tot = {'loss': 0.0, 'hm': 0.0, 'wh': 0.0, 'off': 0.0, 'n': 0}
    model.train() if train else model.eval()
    for xs, boxes_list, _, _ in loader:
        xs = xs.to(device, non_blocking=True)
        B = xs.shape[0]
        with torch.autocast('cuda', enabled=train):
            out, _ = model(xs)
            hm_t, wh_t, off_t, ind, mask = [], [], [], [], []
            for b in boxes_list:
                t = build_targets(b.tolist(), xs.shape[2], xs.shape[3], stride, device=device)
                hm_t.append(t[0]); wh_t.append(t[1]); off_t.append(t[2]); ind.append(t[3]); mask.append(t[4])
            hm_t = torch.stack(hm_t); wh_t = torch.stack(wh_t); off_t = torch.stack(off_t)
            ind = torch.stack(ind); mask = torch.stack(mask)
            l_hm = focal_loss(out['hm'], hm_t)
            l_wh = reg_l1(out['wh'], ind, mask, wh_t)
            l_off = reg_l1(out['off'], ind, mask, off_t)
            loss = l_hm + 0.1 * l_wh + l_off
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        tot['loss'] += float(loss) * B; tot['hm'] += float(l_hm) * B
        tot['wh'] += float(l_wh) * B; tot['off'] += float(l_off) * B; tot['n'] += B
    return {k: v / max(tot['n'], 1) for k, v in tot.items() if k != 'n'}


@torch.no_grad()
def evaluate(model, loader, device, stride=4, thresh=0.2, max_batches=None):
    model.eval()
    preds, gts, attrs_all = [], [], []
    for bi, (xs, boxes_list, _, attrs) in enumerate(loader):
        xs = xs.to(device)
        out, _ = model(xs)
        dets = decode({k: v.float() for k, v in out.items()}, stride=stride, thresh=thresh)
        for b, bb in enumerate(boxes_list):
            xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2,
                                bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2], 1).numpy()
            gts.append(xyxy)
            preds.append(np.array(dets[b]) if dets[b] else np.zeros((0, 5)))
            attrs_all.append(attrs[b])
        if max_batches and bi + 1 >= max_batches:
            break
    return preds, gts, attrs_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:/event-camera/DARPA FENCE/data/sted')
    ap.add_argument('--channel', default='event', choices=['frame', 'event', 'fusion'])
    ap.add_argument('--recurrent', action='store_true')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--bins', type=int, default=5)
    ap.add_argument('--tag', default='')
    ap.add_argument('--chunk', type=int, default=8)
    ap.add_argument('--max_train_seqs', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cin = {'frame': 1, 'event': args.bins, 'fusion': 1 + args.bins}[args.channel]
    if args.recurrent:
        ds_tr = StedChunkDataset(args.data, 'train', args.channel, bins=args.bins,
                                 train=True, chunk=args.chunk,
                                 max_seqs=args.max_train_seqs or None)
        ds_va = StedChunkDataset(args.data, 'val', args.channel, bins=args.bins, chunk=args.chunk)
        ld_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=0,
                           collate_fn=collate_chunk, pin_memory=True)
        ld_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=0,
                           collate_fn=collate_chunk, pin_memory=True)
    else:
        ds_tr = StedDataset(args.data, 'train', args.channel, bins=args.bins, train=True,
                            max_seqs=args.max_train_seqs or None)
        ds_va = StedDataset(args.data, 'val', args.channel, bins=args.bins)
        ld_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=0,
                           collate_fn=collate, pin_memory=True)
        ld_va = DataLoader(ds_va, batch_size=args.bs, shuffle=False, num_workers=0,
                           collate_fn=collate, pin_memory=True)
    model = CenterNet(cin=cin, width=args.width, recurrent=args.recurrent).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'model params: {nparam/1e6:.2f}M | channel {args.channel} | recurrent {args.recurrent}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(ld_tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    scaler = torch.amp.GradScaler('cuda')

    run_name = f'{args.channel}{"_r" if args.recurrent else ""}{args.tag}'
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', run_name)
    os.makedirs(out_dir, exist_ok=True)
    log = []
    best = 0.0
    for ep in range(args.epochs):
        t0 = time.time()
        if args.recurrent:
            tr = run_epoch_seq(model, ld_tr, opt, scaler, device)
            sched.step()
            preds, gts, _ = evaluate_seq(model, ld_va, device)
        else:
            tr = run_epoch(model, ld_tr, opt, scaler, device, train=True)
            sched.step()
            preds, gts, _ = evaluate(model, ld_va, device, max_batches=None)
        r = compute_map(preds, gts)
        dt = time.time() - t0
        print(f'ep {ep}: loss {tr["loss"]:.4f} (hm {tr["hm"]:.4f} wh {tr["wh"]:.3f} off {tr["off"]:.3f}) '
              f'| val mAP {r["mAP"]:.4f} mAP50 {r["mAP50"]:.4f} | {dt:.0f}s', flush=True)
        log.append(dict(epoch=ep, **{f'tr_{k}': v for k, v in tr.items()},
                        val_mAP=r['mAP'], val_mAP50=r['mAP50'], secs=dt))
        if r['mAP50'] > best or ep == 0:
            best = r['mAP50']
            torch.save(dict(model=model.state_dict(), args=vars(args)),
                       os.path.join(out_dir, 'best.pt'))
    json.dump(log, open(os.path.join(out_dir, 'log.json'), 'w'), indent=1)
    print('best mAP50:', round(best, 4), '| saved to', out_dir)


if __name__ == '__main__':
    main()
