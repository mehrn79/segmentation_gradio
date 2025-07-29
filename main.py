from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api_service.api import endpoints
from configs.app_config import AppConfig  

app = FastAPI(
    title="Medical AI Segmentation API",
    description="segmentation and annotation apis",
    version="1.0.0"
)

MASKS_DIR = "/media/external20/mehran_advand/segmentation_gradio/temp"  

app.mount("/static-masks", StaticFiles(directory=MASKS_DIR), name="static-masks")

app.include_router(endpoints.router)
