import gradio as gr
from ct_process import precompute_slices
from segmentation_action import action
from overlay_mask import overlay_masks
import os

target_organs = ["liver", "spleen", "kidney_right", "kidney_left", "gallbladder", "stomach", "pancreas"]

def get_slice(slice_files, slice_idx, selected_organs, patient_id):
    if slice_files is None:
        return None, slice_idx

    ct_slice_path = slice_files[slice_idx]

    if not selected_organs or not patient_id:
        return ct_slice_path, slice_idx

    out_img = overlay_masks(ct_slice_path, selected_organs, patient_id, slice_idx)
    return out_img, slice_idx

def run_segmentation(selected_organs, file):
    if not selected_organs:
        return "⚠️ Please select at least one organ!", gr.update(value=None)

    if file is None:
        return "⚠️ Please upload a NIfTI file!", gr.update(value=None)

    nii_path = file.name
    patient_id = os.path.splitext(os.path.basename(nii_path))[0]

    action(nii_path, selected_organs)

    return f"✅ Segmentation started for: {', '.join(selected_organs)}", patient_id


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 CT Abdomen Viewer + Segmentation Overlay")

    with gr.Row():
        # 📁 ستون چپ
        with gr.Column(scale=3):
            file_input = gr.File(label="Upload NIfTI (.nii.gz)")

            with gr.Accordion("Segmentation", open=True):
                organs_checkbox = gr.CheckboxGroup(choices=target_organs, label="Select target organs")
                segment_button = gr.Button("Run Segmentation")
                output_text = gr.Textbox(label="Status")

            # patient ID و انتخاب ارگان‌ها را در State ذخیره می‌کنیم
            selected_organs_state = gr.State()
            patient_id_state = gr.State()

            segment_button.click(
                run_segmentation,
                inputs=[organs_checkbox, file_input],
                outputs=[output_text, patient_id_state]
            )

            # وقتی سگمنتیشن اجرا شد، State ارگان‌ها را هم آپدیت کن
            segment_button.click(
                lambda organs: organs,
                inputs=organs_checkbox,
                outputs=selected_organs_state
            )

        # 📷 ستون راست: CT Viewer + Overlay
        with gr.Column(scale=7):
            img_out = gr.Image(label="CT Slice + Masks")
            slice_slider = gr.Slider(0, 100, step=1, value=0, label="Slice", visible=True)
            slice_files = gr.State()

            # آپلود CT ➜ کش اسلایس‌ها
            file_input.upload(precompute_slices, inputs=file_input, outputs=[slice_slider, slice_files, slice_slider])

            # وقتی اسلایدر تغییر کند ➜ Overlay بساز
            slice_slider.change(
                get_slice,
                inputs=[slice_files, slice_slider, selected_organs_state, patient_id_state],
                outputs=[img_out, slice_slider]
            )

            file_input.upload(
                lambda slice_files, slice_idx: (slice_files[slice_idx], slice_idx)
                if slice_files else (None, slice_idx),
                inputs=[slice_files, slice_slider],
                outputs=[img_out, slice_slider]
            )
demo.launch()