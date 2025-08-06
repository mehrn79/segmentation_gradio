from pathlib import Path
import uuid
from configs.app_config import AppConfig
from utils.nifti import prepare_nifti_slices
from utils.annotation import handle_annotation_and_segmentation
import shutil
from PIL import Image


def run_annotation_segmentation(tool: str, slice_idx: int, brush_np, box_data, file: Path):
    session_root = AppConfig.BASE_OUTPUT_DIR
    session_path = session_root / f"session_{uuid.uuid4().hex}"
    uploaded_files_dir = session_path / "uploaded_files"
    temp_slices_dir = session_path / "temp_slices"
    annotation_path = session_path / "annotations"

    uploaded_files_dir.mkdir(parents=True, exist_ok=True)
    temp_slices_dir.mkdir(parents=True, exist_ok=True)
    annotation_path.mkdir(parents=True, exist_ok=True)

    uploaded_file_path = uploaded_files_dir / file.name
    shutil.copy(str(file), str(uploaded_file_path))

    prepare_nifti_slices(uploaded_file_path, temp_slices_dir)

    patient_id = uploaded_file_path.name.split('.')[0]

    formatted_box_data = None
    if box_data:
        formatted_box_data = {
            "image": None,
            "boxes": box_data
        }

    brush_mask_path = None
    if brush_np is not None:
        brush_mask_path = annotation_path / f"{patient_id}_slice_{slice_idx:03d}_mask.png"
        Image.fromarray(brush_np).save(brush_mask_path)

    handle_annotation_and_segmentation(
        tool=tool,
        annotator_box_data=formatted_box_data,
        brush_data=brush_mask_path,
        slice_idx=slice_idx,
        patient_id=patient_id,
        session_path_str=session_path,
        ct_file=file
    )

    return "✅ Annotation & Segmentation completed", str(session_path / "medsam2_outputs/png_masks"), patient_id
