import os
import sys

medsam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MedSAM2'))
sys.path.append(medsam2_path)

from sam2.build_sam import build_sam2_video_predictor_npz

import inspect
cfg_path = "configs/sam2.1_hiera_t512.yaml"
ckpt_path = "/media/external20/mehran_advand/segmentation_gradio/MedSAM2/checkpoints/MedSAM2_CTLesion.pt"

predictor = build_sam2_video_predictor_npz(cfg_path, ckpt_path)

sig = inspect.signature(predictor.add_new_points_or_box)
print(f"Signature: {sig}")

