"""Fetch 10,000 COCO train2017 images (seeded random sample, disjoint from the 500 calibration images) for fine-tuning,
plus their YOLO-format labels (read from Day 1's labels.zip, read-only).

    python scripts/prepare_train_data.py  -> data/train2017/images, data/train2017/labels, data/train2017/list.txt
"""
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DAY2 = Path(__file__).resolve().parent.parent
LABELS_ZIP = DAY2.parent / "Day 1" / "data" / "coco" / "labels.zip"
OUT = DAY2 / "data" / "train2017"
N, SEED = 10000, 1

(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(LABELS_ZIP)
lab = {Path(n).stem: n for n in z.namelist() if "labels/train2017/" in n and n.endswith(".txt")}
calib = set((DAY2 / "data" / "calib_train2017" / "list.txt").read_text().split())
names = sorted(s for s in lab if s not in calib)
random.Random(SEED).shuffle(names)
names = names[:N]


def fetch(stem):
    dst = OUT / "images" / f"{stem}.jpg"
    if not (dst.exists() and dst.stat().st_size > 0):
        try:
            urllib.request.urlretrieve(f"http://images.cocodataset.org/train2017/{stem}.jpg", dst)
        except Exception as e:
            print("failed", stem, e)
            return False
    (OUT / "labels" / f"{stem}.txt").write_bytes(z.read(lab[stem]))
    return True


with ThreadPoolExecutor(8) as ex:
    ok = sum(ex.map(fetch, names))
(OUT / "list.txt").write_text("\n".join(names))
print(f"{ok}/{N} training images in {OUT / 'images'}")
