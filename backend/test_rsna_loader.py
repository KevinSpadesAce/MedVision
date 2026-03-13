import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw


LABEL_CSV = r"E:/Documents/GitHub/MedVision/data/stage_2_train_labels.csv"
TRAIN_IMAGE_DIR = r"E:/Documents/GitHub/MedVision/data/stage_2_train_images"
OUTPUT_DIR = Path(r"E:/Documents/GitHub/MedVision/backend/rsna_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def dicom_to_pil(dicom_path: str) -> Image.Image:
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)

    image = image - image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val

    image = (image * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def main():
    df = pd.read_csv(LABEL_CSV)

    print("CSV loaded.")
    print("Total rows:", len(df))
    print("Columns:", list(df.columns))

    positive_df = df[df["Target"] == 1]
    print("Positive rows:", len(positive_df))

    if positive_df.empty:
        raise ValueError("No positive samples found in labels CSV.")

    # 取第一张有框的样本
    patient_id = positive_df.iloc[0]["patientId"]
    patient_rows = positive_df[positive_df["patientId"] == patient_id]

    dicom_path = os.path.join(TRAIN_IMAGE_DIR, f"{patient_id}.dcm")
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOM not found: {dicom_path}")

    image = dicom_to_pil(dicom_path)
    draw = ImageDraw.Draw(image)

    findings = []

    for _, row in patient_rows.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])
        bbox = [x, y, x + w, y + h]

        draw.rectangle(bbox, outline="red", width=4)
        draw.text((x, max(0, y - 18)), "pneumonia", fill="red")

        findings.append({
            "label": "pneumonia",
            "confidence": 1.0,
            "bbox": bbox
        })

    out_path = OUTPUT_DIR / f"{patient_id}.png"
    image.save(out_path)

    print("Patient ID:", patient_id)
    print("Findings:", findings)
    print("Saved annotated image to:", out_path)


if __name__ == "__main__":
    main()