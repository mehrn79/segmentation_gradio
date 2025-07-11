import shutil
import gradio as gr
from pathlib import Path
from datetime import datetime
import uuid

from configs.app_config import AppConfig
from utils.image import create_overlay_image
from segmentation import segment
from utils.nifti import prepare_nifti_slices


def handle_file_upload(file):
    if file is None:
        return None, None, None, None, None

    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session_path = AppConfig.BASE_OUTPUT_DIR / session_id

    uploaded_files_dir = session_path / "uploaded_files"
    temp_slices_dir = session_path / "temp_slices"

    uploaded_files_dir.mkdir(parents=True, exist_ok=True)
    temp_slices_dir.mkdir(parents=True, exist_ok=True)

    uploaded_file_path = uploaded_files_dir / Path(file.name).name
    shutil.copy(str(Path(file.name)), str(uploaded_file_path))

    slice_files, num_slices = prepare_nifti_slices(
        uploaded_file_path, temp_slices_dir)

    if not slice_files:
        return None, None, None, None, None

    initial_slice_idx = num_slices // 2
    patient_id = uploaded_file_path.name.split('.')[0]

    initial_image = create_overlay_image(
        slice_files[initial_slice_idx], [], patient_id, initial_slice_idx, str(session_path))

    return (
        gr.update(visible=True, maximum=num_slices - 1, value=initial_slice_idx),
        slice_files,
        initial_image,
        patient_id,
        str(session_path)
    )


def run_segmentation(selected_organs, file, session_path_str: str):
    if not selected_organs:
        return "⚠️ Please select at least one organ.", gr.update(), gr.update()
    if file is None:
        return "⚠️ Please upload a NIfTI file first.", gr.update(), gr.update()

    session_path = Path(session_path_str)
    nii_path = session_path / "uploaded_files" / Path(file.name).name

    if not nii_path.exists():
        return f"Error: Uploaded file not found at {nii_path}", gr.update(), gr.update()

    status_message = segment(nii_path, selected_organs, session_path)
    patient_id = nii_path.name.split('.')[0]

    return status_message, selected_organs, patient_id


def update_slice_view(slice_idx, slice_files, selected_organs, patient_id, session_path_str: str):
    if not slice_files or patient_id is None or not session_path_str:
        return None

    return create_overlay_image(slice_files[slice_idx], selected_organs, patient_id, slice_idx, session_path_str)


if __name__ == "__main__":
    AppConfig.setup_directories()
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("## 🧠 CT Abdomen Viewer & Segmentation")

        slice_files_state = gr.State()
        patient_id_state = gr.State()
        selected_organs_state = gr.State([])
        session_path_state = gr.State()

        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(
                    label="Upload NIfTI file (.nii or .nii.gz)")

                with gr.Accordion("Segmentation Controls", open=True):
                    organs_checkbox = gr.CheckboxGroup(
                        choices=AppConfig.TARGET_ORGANS, label="Select Target Organs")
                    segment_button = gr.Button(
                        "▶️ Run Segmentation", variant="primary")

                output_text = gr.Textbox(
                    label="Pipeline Status", interactive=False)

            with gr.Column(scale=7):
                img_out = gr.Image(
                    label="CT Slice with Mask Overlay", type="numpy", height=600)
                slice_slider = gr.Slider(
                    minimum=0, maximum=100, step=1, label="Slice", visible=False)

        outputs = [slice_slider, slice_files_state,
                   img_out, patient_id_state, session_path_state]
        file_input.upload(
            fn=handle_file_upload,
            inputs=[file_input],
            outputs=outputs
        )

        segment_button.click(
            fn=run_segmentation,
            inputs=[organs_checkbox, file_input, session_path_state],
            outputs=[output_text, selected_organs_state, patient_id_state]
        ).then(
            fn=update_slice_view,
            inputs=[slice_slider, slice_files_state,
                    selected_organs_state, patient_id_state, session_path_state],
            outputs=[img_out]
        )

        interactive_inputs = [slice_slider, slice_files_state,
                              selected_organs_state, patient_id_state, session_path_state]
        slice_slider.change(fn=update_slice_view,
                            inputs=interactive_inputs, outputs=[img_out])
        organs_checkbox.change(fn=update_slice_view,
                               inputs=interactive_inputs, outputs=[img_out])
    demo.launch()
