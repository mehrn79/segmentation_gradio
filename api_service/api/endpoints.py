from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil
from api_service.models.organ_segmentation import run_segmentation
from api_service.models.medsam2 import run_annotation_segmentation
import numpy as np
from PIL import Image
import io
import json

router = APIRouter()

@router.post("/segment")
def segmentation_endpoint(file: UploadFile = File(...)):
    temp_path = Path("/tmp/uploads")
    temp_path.mkdir(parents=True, exist_ok=True)
    file_path = temp_path / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    message, png_masks_dir, patient_id = run_segmentation(file_path)
    return {
        "message": message,
        "output_dir": png_masks_dir,
        "patient_id": patient_id
    }



@router.post("/annotate-segment")
def annotation_segmentation_endpoint(
    file: UploadFile = File(...),
    image: UploadFile = File(None),
    box: str = Form(None),
    slice_idx: int = Form(...),
    tool: str = Form(...)
):
    temp_path = Path("/tmp/uploads")
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

    return {
        "message": message,
        "output_dir": png_masks_dir,
        "patient_id": patient_id
    }
