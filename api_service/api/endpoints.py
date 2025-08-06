from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil
from api_service.models.organ_segmentation import run_segmentation
from api_service.models.medsam2 import run_annotation_segmentation
import numpy as np
from PIL import Image
import io
import json

from api_service.schemas.segment_schemas import SegmentationResponse
from configs.app_config import AppConfig
from api_service.schemas.annotation_segment_docs import ANNOTATE_SEGMENT_DESCRIPTION
from api_service.schemas.segment_docs import SEGMENT_ENDPOINT_DESCRIPTION
from utils.mask import load_masks_as_base64, load_flat_masks_as_base64

router = APIRouter()


@router.post(
    "/segment",
    summary="Segmentation Endpoint",
    description=SEGMENT_ENDPOINT_DESCRIPTION,
    response_model=SegmentationResponse
)
def segmentation_endpoint(file: UploadFile = File(...)):
    temp_path = AppConfig.TEMP_UPLOAD_DIR
    temp_path.mkdir(parents=True, exist_ok=True)
    file_path = temp_path / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    message, png_masks_dir, patient_id = run_segmentation(file_path)

    masks = load_masks_as_base64(Path(png_masks_dir))
    shutil.rmtree(Path(png_masks_dir).parent, ignore_errors=True)

    return {
        "message": message,
        "patient_id": patient_id,
        "masks": masks
    }


@router.post(
    "/annotate-segment",
    summary="Annotation-based Segmentation with MedSAM2",
    description=ANNOTATE_SEGMENT_DESCRIPTION
)
def annotation_segmentation_endpoint(
    file: UploadFile = File(...),
    image: UploadFile = File(None),
    box: str = Form(None),
    slice_idx: int = Form(...),
    tool: str = Form(...)
):
    temp_path = AppConfig.TEMP_UPLOAD_DIR
    temp_path.mkdir(parents=True, exist_ok=True)
    file_path = temp_path / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_np = None
    if image is not None:
        image_path = temp_path / image.filename
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        brush_image = Image.open(image_path).convert("RGBA")
        image_np = np.array(brush_image)

    box_data = None
    if box is not None:
        try:
            box_data = json.loads(box)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON format in box."}

    message, png_masks_dir, patient_id = run_annotation_segmentation(
        tool, slice_idx, image_np, box_data, file_path
    )

    masks = load_flat_masks_as_base64(Path(png_masks_dir))
    shutil.rmtree(Path(png_masks_dir).parent.parent, ignore_errors=True)

    return {
        "message": message,
        "patient_id": patient_id,
        "masks": masks
    }
