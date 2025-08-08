import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from api_service.models.organ_segmentation import run_segmentation
from api_service.models.medsam2 import run_annotation_segmentation
from configs.app_config import AppConfig

from utils.nifti import create_png_masks_from_nifti
from utils.mask import load_flat_masks_as_base64, load_masks_as_base64


router = APIRouter()


@router.post(
    "/segment",
    summary="Segmentation Endpoint",
    response_class=JSONResponse
)
def segmentation_endpoint(file: UploadFile = File(...)):
    # file_path = AppConfig.TEMP_UPLOAD_DIR / file.filename
    # with file_path.open("wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)

    # segmentation_output_dir, patient_id, session_path = run_segmentation(
    #     file_path)

    # png_masks_dir = session_path / "organ_masks"
    # png_masks_dir.mkdir(parents=True, exist_ok=True)

    # create_png_masks_from_nifti(segmentation_output_dir, png_masks_dir)

    masks = load_masks_as_base64(Path(
        # (png_masks_dir)
        '/Users/reza/Projects/segmentation_gradio/output/session_61f3f760b2524d879bcab737ee4f2a62/organ_masks'))

    return JSONResponse(content={
        # "patient_id": patient_id,
        "masks": masks
    })


@router.post(
    "/annotate-segment",
    summary="Annotation-based Segmentation with MedSAM2",
)
def annotation_segmentation_endpoint(
    file: UploadFile = File(...),
    slice_idx: int = Form(...),
    image: UploadFile = File(None),
    box: str = Form(None)
):
    tool = None
    box_data = None
    image_np = None

    if box:
        tool = "Bounding Box"
        try:
            box_data = json.loads(box)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Invalid JSON format in 'box'.")
    elif image:
        tool = "Brush"

    if not tool:
        raise HTTPException(
            status_code=400, detail="Either 'box' or 'image' (for brush mask) must be provided.")

    file_path = AppConfig.TEMP_UPLOAD_DIR / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if image is not None:
        image_path = AppConfig.TEMP_UPLOAD_DIR / image.filename
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        brush_image = Image.open(image_path).convert("RGBA")
        image_np = np.array(brush_image)

    try:
        png_masks_dir, patient_id = run_annotation_segmentation(
            tool=tool,
            slice_idx=slice_idx,
            brush_np=image_np,
            box_data=box_data,
            file=file_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    masks = load_flat_masks_as_base64(Path(png_masks_dir))

    return {
        "patient_id": patient_id,
        "masks": masks
    }
