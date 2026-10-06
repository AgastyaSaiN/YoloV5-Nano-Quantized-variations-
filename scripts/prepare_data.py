"""Download COCO val2017 + YOLO-format labels, then split into a calibration set and a held-out eval set.

calib: first 300 images (only the first 100 are used - see docs/METHOD.md)
eval : last 1000 images (never seen during calibration), with labels
"""
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "coco"
DATA.mkdir(parents=True, exist_ok=True)
URLS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",                                   # ~780 MB
    "labels.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",  # ~46 MB
}
for fname, url in URLS.items():
    if not (DATA / fname).exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, DATA / fname)

z = zipfile.ZipFile(DATA / "val2017.zip")
names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
lz = zipfile.ZipFile(DATA / "labels.zip")
labels = {Path(n).name: n for n in lz.namelist() if "labels/val2017/" in n and n.endswith(".txt")}

for d in ["calib", "eval/images", "eval/labels"]:
    (DATA / d).mkdir(parents=True, exist_ok=True)
for n in names[:300]:
    (DATA / "calib" / Path(n).name).write_bytes(z.read(n))
for n in names[-1000:]:
    b = Path(n).name
    (DATA / "eval/images" / b).write_bytes(z.read(n))
    t = b.replace(".jpg", ".txt")
    (DATA / "eval/labels" / t).write_bytes(lz.read(labels[t]) if t in labels else b"")

class_lines = "".join(f"  {i}: c{i}\n" for i in range(80))
(DATA / "eval.yaml").write_text(f"path: {(DATA / 'eval').as_posix()}\ntrain: images\nval: images\nnames:\n{class_lines}")
print("calib:", len(list((DATA / 'calib').iterdir())), "eval:", len(list((DATA / 'eval/images').iterdir())))
