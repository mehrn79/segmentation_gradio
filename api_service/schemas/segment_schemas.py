from pydantic import BaseModel, Field
from typing import Dict

class SegmentationRequest(BaseModel):
    session_path: str = Field(..., example="/path/to/session")

class SegmentationResponse(BaseModel):
    message: str
    patient_id: str
    masks: Dict[str, Dict[str, Dict[str, str]]]  # patient → organ → filename → base64
