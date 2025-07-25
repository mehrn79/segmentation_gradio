import logging
from pathlib import Path
from monai.bundle import ConfigParser

from configs.app_config import AppConfig
from utils.nifti import convert_nifti_to_dicom, create_png_masks_from_nifti


def setup_session_directories(session_path: Path) -> tuple[Path, Path, Path]:
    session_dicom_dir = session_path / "Images_dicom"
    session_segmentation_output_dir = session_path / "segmentation_output"
    session_png_masks_dir = session_path / "organ_masks"

    session_dicom_dir.mkdir(parents=True, exist_ok=True)
    session_segmentation_output_dir.mkdir(parents=True, exist_ok=True)
    session_png_masks_dir.mkdir(parents=True, exist_ok=True)

    return session_dicom_dir, session_segmentation_output_dir, session_png_masks_dir


def generate_monai_config(nii_path: Path, session_path: Path, output_dir: Path) -> Path:
    logging.info("Creating dynamic MONAI configuration...")
    parser = ConfigParser()
    parser.read_config(AppConfig.MONAI_CONFIG_PATH)

    parser["dataset_dir"] = str(nii_path.parent)
    parser["output_dir"] = str(output_dir)

    temp_monai_config_path = session_path / "monai-inference-temp.json"
    parser.export_config_file(parser.config, temp_monai_config_path)

    if not temp_monai_config_path.exists():
        logging.error(
            f"Failed to create temporary MONAI config file: {temp_monai_config_path}")
        return None

    return temp_monai_config_path


def execute_monai_segmentation(config_path: Path):
    if not config_path:
        logging.error("MONAI config path is not provided!!!")
        return

    logging.info("Initializing and running MONAI evaluator...")
    evaluator_parser = ConfigParser()
    evaluator_parser.read_config(config_path)
    evaluator = evaluator_parser.get_parsed_content("evaluator")
    evaluator.run()
    logging.info("MONAI evaluation complete.")


def segment(nii_path: Path, selected_organs: list, session_path: Path) -> str:
    session_dicom_dir, seg_output_dir, png_masks_dir = setup_session_directories(
        session_path)

    convert_nifti_to_dicom(nii_path, session_dicom_dir)

    temp_monai_config = generate_monai_config(
        nii_path, session_path, seg_output_dir)

    execute_monai_segmentation(temp_monai_config)

    create_png_masks_from_nifti(
        seg_output_dir, selected_organs, png_masks_dir)

    return f"✅ Segmentation complete for: {', '.join(selected_organs)}"
