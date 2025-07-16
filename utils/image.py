from configs.app_config import AppConfig
import numpy as np
import cv2
import logging
from pathlib import Path


def apply_ct_window(image: np.ndarray, level: int = 40, width: int = 400) -> np.ndarray:
    lower = level - (width / 2)
    upper = level + (width / 2)
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return (np.clip(windowed, 0, 1) * 255).astype(np.uint8)


def create_overlay_image(ct_slice_path: str, selected_organs: list, patient_id: str, slice_idx: int, session_path_str: str) -> np.ndarray:
    ct_img = cv2.imread(ct_slice_path, cv2.IMREAD_GRAYSCALE)
    if ct_img is None:
        logging.warning(f"Could not read CT slice: {ct_slice_path}")
        return np.zeros((512, 512, 3), dtype=np.uint8)

    overlay = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2BGR)

    if not selected_organs or not patient_id or not session_path_str:
        return overlay

    session_path = Path(session_path_str)
    png_masks_dir = session_path / "organ_masks"

    for organ in selected_organs:
        mask_path = png_masks_dir / patient_id / \
            organ / f'slice_{slice_idx:03d}_OUT.png'
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning(f"Failed to load mask: {mask_path}")
            continue

        color = AppConfig.ORGAN_COLORS.get(organ, (255, 255, 255))
        mask_bool = mask > 0

        colored_mask = np.zeros_like(overlay)
        colored_mask[mask_bool] = color
        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool], 0.5, colored_mask[mask_bool], 0.5, 0)

    return overlay
