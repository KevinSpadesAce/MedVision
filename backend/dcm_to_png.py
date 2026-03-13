import sys
import numpy as np
import pydicom
from PIL import Image

dcm_path = sys.argv[1]
png_path = sys.argv[2]

ds = pydicom.dcmread(dcm_path)
img = ds.pixel_array.astype(np.float32)

img = img - img.min()
if img.max() > 0:
    img = img / img.max()

img = (img * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(img).save(png_path)

print(f"Saved PNG to: {png_path}")