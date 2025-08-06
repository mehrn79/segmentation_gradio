from pathlib import Path
import uuid
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from segmentation import segment

def run_segmentation(file_path: Path):
    # Create session output path automatically
    session_root = Path("/tmp/segmentation_sessions")
    session_root.mkdir(parents=True, exist_ok=True)
    session_path = session_root / f"session_{uuid.uuid4().hex}"
    session_path.mkdir(parents=True, exist_ok=True)

    png_masks_dir = segment(file_path, session_path)
    patient_id = file_path.name.split('.')[0]

    return "✅ Segmentation completed", str(png_masks_dir), patient_id