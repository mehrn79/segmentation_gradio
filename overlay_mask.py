import cv2
import numpy as np
import os

organ_colors = {
    "liver": (0, 0, 255),      
    "spleen": (0, 255, 0),     
    "kidney_right": (255, 0, 0),
    "kidney_left": (255, 255, 0),
    "gallbladder": (255, 0, 255),
    "stomach": (0, 255, 255),
    "pancreas": (128, 0, 128)
}

def overlay_masks(ct_slice_path, selected_organs, patient_id, slice_idx):
    # چک کن که CT Slice وجود داشته باشه
    ct_img = cv2.imread(ct_slice_path, cv2.IMREAD_GRAYSCALE)
    if ct_img is None:
        raise ValueError(f"❌ CT Slice not found: {ct_slice_path}")

    ct_img_rgb = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2BGR)

    for organ in selected_organs:
        mask_path = f'organ_masks/{patient_id.split('.')[0]}/{organ}/slice_{slice_idx:03d}_OUT.png'
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Mask not found or failed to load: {mask_path}")
            continue

        color = organ_colors.get(organ, (255, 255, 255))

        mask_colored = np.zeros_like(ct_img_rgb)
        for c in range(3):
            mask_colored[:, :, c] = mask / 255 * color[c]

        alpha = 0.5
        mask_bool = mask > 0

        if np.any(mask_bool):  # فقط اگه mask غیر صفر داره
            ct_img_rgb[mask_bool] = cv2.addWeighted(
                ct_img_rgb[mask_bool], 1 - alpha,
                mask_colored[mask_bool], alpha,
                0
            )

    out_path = "temp_overlay.png"
    cv2.imwrite(out_path, ct_img_rgb)
    return out_path
