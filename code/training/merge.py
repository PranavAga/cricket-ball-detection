# DATASET UNIFICATION & LABEL CLEANING SCRIPT
# This script:
# 1. Merges multiple YOLO datasets into one
# 2. Filters labels to keep ONLY the cricket ball
# 3. Remaps all valid detections to class_id = 0
# 4. Creates a unified train/val/test structure

import shutil
from pathlib import Path


# Root directory where original datasets live
DATASETS = {
    "dataset1": {
        "path": Path("datasets/Ball Detection Cricket.v1i.yolov11"),
        "ball_class_ids": [2],
    },
    "dataset2": {
        "path": Path("datasets/Cricket ball detection.v1i.yolov11"),
        "ball_class_ids": [0],
    },
    "dataset3": {
        "path": Path("datasets/Cricket ball.v1-roboflow"),
        "ball_class_ids": [0],
    },
}

# Unified dataset output
OUT_ROOT = Path("datasets/cricket_ball")
SPLITS = ["train", "valid", "test"]


def prepare_dirs():
    for split in SPLITS:
        (OUT_ROOT / split / "images" ).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def process_dataset(ds_cfg):
    base = ds_cfg["path"]
    ball_ids = ds_cfg["ball_class_ids"]

    for split in SPLITS:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"

        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*"):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            new_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    if cls_id in ball_ids:
                        parts[0] = "0"  # remap to single class
                        new_lines.append(" ".join(parts))

            if not new_lines:
                continue  # skip images without balls

            # copy image
            shutil.copy2(img_path, OUT_ROOT / split  / "images" / img_path.name)

            # write cleaned label
            with open(OUT_ROOT / split  / "labels" / f"{img_path.stem}.txt", "w") as f:
                f.write("\n".join(new_lines))


def main():
    prepare_dirs()
    for ds in DATASETS.values():
        process_dataset(ds)
    print("Unified cricket ball dataset created successfully.")


if __name__ == "__main__":
    main()
