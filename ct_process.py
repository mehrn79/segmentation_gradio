import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import gradio as gr

def apply_window(image, level=40, width=400):
    lower = level - (width / 2)
    upper = level + (width / 2)
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    windowed = np.clip(windowed, 0, 1)
    return windowed

def precompute_slices(file):
    # 1️⃣ مسیر ذخیره فایل
    os.makedirs("uploaded_files", exist_ok=True)
    save_path = os.path.join("uploaded_files", os.path.basename(file.name))
    shutil.copy(file.name, save_path)

    img = nib.load(file.name)
    data = img.get_fdata()
    num_slices = data.shape[2]

    slice_files = []
    temp_dir = "temp_slices"
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(num_slices):
        slice_img = apply_window(data[:, :, i])
        fname = os.path.join(temp_dir, f"slice_{i}.png")
        plt.imsave(fname, slice_img, cmap='gray')
        slice_files.append(fname)

    return (
        gr.update(visible=True, maximum=num_slices - 1, value=num_slices // 2),
        slice_files,
        num_slices // 2
    )