from pathlib import Path
import subprocess
import subprocess
from configs.app_config import AppConfig

BASE_DIR = Path(__file__).resolve().parents[1]

def run_medsam2_prediction(ct_path, slice_idx, session_path_str, patient_id, tool, annotation_path, boxes_data):
    output_path = Path(session_path_str) / "medsam2_outputs"
    output_path.mkdir(parents=True, exist_ok=True)
    output_mask_path = output_path / f"{patient_id}_pred_mask.nii.gz"

    if tool == "Bounding Box":
        try:
            box_obj = boxes_data["boxes"]
            x1 = box_obj["xmin"]
            y1 = box_obj["ymin"]
            x2 = box_obj["xmax"]
            y2 = box_obj["ymax"]
            box_str = f"{x1} {y1} {x2} {y2}"

        except Exception as e:
            return f"❌ Failed to parse bounding box data: {e}", None, None

        cmd = [
            "python", f"{BASE_DIR}/utils/med_sam.py",
            "--ct_path", str(ct_path),
            "--key_slice_idx", str(slice_idx),
            "--box", *box_str.split(),
            "--save_path", str(output_mask_path),
            "--checkpoint", str(AppConfig.MEDSAM_CHECKPOINT_PATH),
            "--cfg", str(AppConfig.MEDSAM_CONFIG_PATH)
        ]

    elif tool == "Brush":
        cmd = [
            "python", f"{BASE_DIR}/utils/med_sam.py",
            "--ct_path", str(ct_path),
            "--key_slice_idx", str(slice_idx),
            "--mask_path", str(annotation_path),
            "--save_path", str(output_mask_path),
            "--checkpoint", str(AppConfig.MEDSAM_CHECKPOINT_PATH),
            "--cfg", str(AppConfig.MEDSAM_CONFIG_PATH)
        ]

    else:
        return "❌ Unknown annotation tool.", None, None

    print("[INFO] Running command:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("[STDOUT]\n", result.stdout.decode())
        print("[STDERR]\n", result.stderr.decode())

        return f"✅ MedSAM2 segmentation completed.\n\n{result.stdout.decode()}", str(output_mask_path), patient_id

    except subprocess.CalledProcessError as e:
        print("[ERROR] Subprocess failed:", e)
        print("[STDERR]\n", e.stderr.decode())
        return f"❌ MedSAM2 failed:\n{e.stderr.decode()}", None, None
    except Exception as ex:
        print("[ERROR] Unexpected failure:", ex)
        return f"❌ Unexpected error: {ex}", None, None

def handle_annotation_and_segmentation(
    tool, annotator_box_data, brush_data,
    slice_idx, patient_id, session_path_str, ct_file
):
    
    slice_name = f"slice_{slice_idx:03d}"
    ann_dir = Path(session_path_str) / "annotations"

    annotation_path = (
        ann_dir / f"{patient_id}_{slice_name}_boxes.json"
        if tool == "Bounding Box"
        else ann_dir / f"{patient_id}_{slice_name}_mask.png"
    )

    seg_status = run_medsam2_prediction(
        ct_path=ct_file,
        session_path_str=session_path_str,
        patient_id=patient_id,
        slice_idx=slice_idx,
        tool=tool,
        annotation_path=brush_data,
        boxes_data = annotator_box_data if tool == "Bounding Box" else None
    )

    return seg_status
