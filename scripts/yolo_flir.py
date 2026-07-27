"""Export FLIR thermal train/val as YOLO-format dataset (frame-channel anchor).

Extracts thermal jpegs from the Kaggle zip and writes YOLO labels for the
6-class benchmark mapping (person,bike,car,motor,bus,truck).
"""
import sys, os, json, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, 'data', 'flir_adas_v2.zip')
OUT = os.path.join(ROOT, 'data', 'flir_yolo')
KEEP = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 8: 5}
NAMES = ['person', 'bike', 'car', 'motor', 'bus', 'truck']


def export_split(z, split, coco):
    img_out = os.path.join(OUT, 'images', split)
    lab_out = os.path.join(OUT, 'labels', split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)
    anns = {}
    for a in coco['annotations']:
        if a['category_id'] in KEEP:
            anns.setdefault(a['image_id'], []).append(a)
    prefix = f'FLIR_ADAS_v2/images_thermal_{split}/'
    n_written = 0
    for im in coco['images']:
        src = prefix + 'data/' + os.path.basename(im['file_name'])
        try:
            data = z.read(src)
        except KeyError:
            try:
                data = z.read(prefix + im['file_name'])
            except KeyError:
                continue
        stem = os.path.splitext(os.path.basename(im['file_name']))[0]
        open(os.path.join(img_out, stem + '.jpg'), 'wb').write(data)
        W, H = im['width'], im['height']
        lines = []
        for a in anns.get(im['id'], []):
            x, y, w, h = a['bbox']
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f'{KEEP[a["category_id"]]} {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}')
        open(os.path.join(lab_out, stem + '.txt'), 'w').write('\n'.join(lines))
        n_written += 1
    print(split, 'written:', n_written)


def main():
    z = zipfile.ZipFile(ZIP)
    for split in ('train', 'val'):
        coco = json.loads(z.read(f'FLIR_ADAS_v2/images_thermal_{split}/coco.json'))
        export_split(z, split, coco)
    yaml = (f'path: {OUT}\n'
            f'train: images/train\nval: images/val\n'
            f'names: {NAMES}\n')
    open(os.path.join(OUT, 'flir_thermal.yaml'), 'w').write(yaml)
    print('yaml written')


if __name__ == '__main__':
    main()
