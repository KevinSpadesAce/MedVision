import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split

# ===== 路径配置 =====
LABEL_CSV = r"E:/Documents/GitHub/MedVision/data/stage_2_train_labels.csv"
DICOM_DIR = r"E:/Documents/GitHub/MedVision/data/stage_2_train_images"

OUTPUT_ROOT = Path(r"E:/Documents/GitHub/MedVision/rsna_yolo")
IMG_TRAIN_DIR = OUTPUT_ROOT / "images" / "train"
IMG_VAL_DIR = OUTPUT_ROOT / "images" / "val"
LBL_TRAIN_DIR = OUTPUT_ROOT / "labels" / "train"
LBL_VAL_DIR = OUTPUT_ROOT / "labels" / "val"

for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== 参数 =====
TOTAL_IMAGES = 1000
POSITIVE_IMAGES = 500
NEGATIVE_IMAGES = 500
RANDOM_SEED = 42


def dicom_to_png(dicom_path: str, png_path: str):
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)

    image = image - image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val

    image = (image * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(image).convert("RGB")
    img.save(png_path)


def bbox_to_yolo(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height


def main():
    random.seed(RANDOM_SEED)

    df = pd.read_csv(LABEL_CSV)

    # 所有 positive image id
    positive_ids = df[df["Target"] == 1]["patientId"].unique().tolist()

    # 所有 image id
    all_ids = df["patientId"].unique().tolist()

    # negative image id：这个 patientId 在 CSV 中只有 Target=0
    positive_id_set = set(positive_ids)
    negative_ids = [pid for pid in all_ids if pid not in positive_id_set]

    print("Total unique images:", len(all_ids))
    print("Positive images:", len(positive_ids))
    print("Negative images:", len(negative_ids))

    sampled_positive = random.sample(positive_ids, min(POSITIVE_IMAGES, len(positive_ids)))
    sampled_negative = random.sample(negative_ids, min(NEGATIVE_IMAGES, len(negative_ids)))

    selected_ids = sampled_positive + sampled_negative
    random.shuffle(selected_ids)

    train_ids, val_ids = train_test_split(
        selected_ids,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    print("Selected total:", len(selected_ids))
    print("Train:", len(train_ids))
    print("Val:", len(val_ids))

    for split_name, split_ids, img_dir, lbl_dir in [
        ("train", train_ids, IMG_TRAIN_DIR, LBL_TRAIN_DIR),
        ("val", val_ids, IMG_VAL_DIR, LBL_VAL_DIR),
    ]:
        print(f"\nProcessing {split_name}...")

        for i, patient_id in enumerate(split_ids, 1):
            dicom_path = os.path.join(DICOM_DIR, f"{patient_id}.dcm")
            png_path = img_dir / f"{patient_id}.png"
            label_path = lbl_dir / f"{patient_id}.txt"

            if not os.path.exists(dicom_path):
                print(f"Missing DICOM: {dicom_path}")
                continue

            # DICOM -> PNG
            dicom_to_png(dicom_path, str(png_path))

            # 获取图像尺寸
            with Image.open(png_path) as img:
                img_w, img_h = img.size

            # 当前 patient 的所有 positive 框
            rows = df[(df["patientId"] == patient_id) & (df["Target"] == 1)]

            yolo_lines = []

            for _, row in rows.iterrows():
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["width"])
                h = float(row["height"])

                x_center, y_center, bw, bh = bbox_to_yolo(x, y, w, h, img_w, img_h)

                # class 0 = pneumonia
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            # negative image 也要创建空 txt
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            if i % 100 == 0:
                print(f"{split_name}: {i}/{len(split_ids)}")

    print("\nDone.")
    print("YOLO dataset prepared at:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()