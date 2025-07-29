ANNOTATE_SEGMENT_DESCRIPTION = """
This endpoint performs medical image segmentation using the MedSAM2 model, with optional annotation input.

🧠 **Use Cases**:
- Upload a CT scan in NIfTI format
- Select a specific slice index
- Choose your annotation method: **Brush** or **Bounding Box**
- Receive a predicted segmentation mask

🔧 **Tools**:
- "Brush": Upload a PNG image with an RGBA brush mask.
- "Bounding Box": Provide bounding box coordinates as a JSON string in the form:
  ```json
  {
    "xmin": 120,
    "ymin": 150,
    "xmax": 220,
    "ymax": 280
  }
  ```

📤 **Required Inputs**:
- `file`: NIfTI file (.nii or .nii.gz)
- `slice_idx`: The index of the CT slice to use for segmentation
- `tool`: Either "Brush" or "Bounding Box"

📥 **Optional Inputs**:
- `image`: PNG mask file (only required if using Brush)
- `box`: JSON-formatted bounding box (only required if using Bounding Box)

📦 **Output**:
A predicted segmentation mask will be saved in the output directory under:
`medsam2_outputs/png_masks/`
"""
