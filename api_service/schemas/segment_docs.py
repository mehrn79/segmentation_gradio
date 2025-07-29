SEGMENT_ENDPOINT_DESCRIPTION = """
This endpoint runs a segmentation model on an uploaded CT scan file.

🧠 **Use Case**:
Upload a NIfTI-format CT scan file (`.nii` or `.nii.gz`).  
The model automatically detects and segments predefined abdominal organs.

📤 **Required Input**:
- `file`: The CT scan file in NIfTI format. This is the primary input to the model.

📦 **Output**:
- A set of predicted segmentation masks will be saved to an output folder.
- Masks are saved in NIfTI and PNG formats under the `masks/` directory inside the session folder.

📁 **Example Output Structure**:
```
session_<timestamp>/
├── uploaded_files/
│   └── your_file.nii.gz
├── masks/
│   └── liver_mask..png
│   └── kidney_mask.png
    ....
```
"""
