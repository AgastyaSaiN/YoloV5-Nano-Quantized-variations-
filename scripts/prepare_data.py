"""Fetch 500 COCO train2017 images (seeded random sample) for calibration.

Calibration images come from train2017 so they never overlap the val2017 test set.
Image names are taken from Day 1's labels.zip (read-only); only images that have labels are used.
"""
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DAY2 = Path(__file__).resolve().parent.parent
LABELS_ZIP = DAY2.parent / "Day 1" / "data" / "coco" / "labels.zip"
OUT = DAY2 / "data" / "calib_train2017"
N, SEED = 500, 0

OUT.mkdir(parents=True, exist_ok=True)
names = sorted(Path(n).stem for n in zipfile.ZipFile(LABELS_ZIP).namelist()
               if "labels/train2017/" in n and n.endswith(".txt"))
random.Random(SEED).shuffle(names)
names = names[:N]


def fetch(stem):
    dst = OUT / f"{stem}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return True
    try:
        urllib.request.urlretrieve(f"http://images.cocodataset.org/train2017/{stem}.jpg", dst)
        return True
    except Exception as e:
        print("failed", stem, e)
        return False


with ThreadPoolExecutor(8) as ex:
    ok = sum(ex.map(fetch, names))
(OUT / "list.txt").write_text("\n".join(names))
print(f"{ok}/{N} calibration images in {OUT}")
