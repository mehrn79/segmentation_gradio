import os
import glob
import nibabel as nib
import numpy as np
import cv2
from natsort import natsorted
from monai_wholeBody_ct_segmentation.organList import *  

def generate_mask(target_organs):
    # مسیرها
    NIFTI_data_dir = '/media/external20/mehran_advand/gradio/Segmentation_Output'
    DCM_data_dir = '/media/external20/mehran_advand/gradio/Images_dicom'
    output_dir = 'organ_masks'

    # ساخت پوشه خروجی در صورت نیاز
    os.makedirs(output_dir, exist_ok=True)

    # لیست بیماران
    patient_folders = natsorted(os.listdir(NIFTI_data_dir))

    for patient_id in patient_folders:

        print(f"\n🧾 Processing patient: {patient_id}")

        # مسیر NIfTI و DICOM
        nii_path = os.path.join(NIFTI_data_dir, patient_id, f"{patient_id}_trans.nii.gz")
        dcm_folder = os.path.join(DCM_data_dir, patient_id)

        if not os.path.isfile(nii_path):
            print(f"❌ NIfTI file not found for {patient_id}")
            continue

        dcm_files = glob.glob(os.path.join(dcm_folder, "*.dcm"))
        if not dcm_files:
            print(f"❌ No DICOM files found for {patient_id}")
            continue

        dcm_files = natsorted(dcm_files)

        # بارگذاری NIfTI
        print(f"📥 Reading NIfTI: {nii_path}")
        nii = nib.load(nii_path)
        label_data = nii.get_fdata()

        # بررسی محور affine و اصلاح در صورت نیاز
        if nii.affine[0, 0] > 0:
            label_data = np.flip(label_data, axis=0)
        if nii.affine[1, 1] > 0:
            label_data = np.flip(label_data, axis=1)
        if nii.affine[2, 2] > 0:
            label_data = np.flip(label_data, axis=2)

        # تغییر محور به Z,Y,X
        label_data = np.transpose(label_data, (2, 1, 0))
        num_slices = min(len(label_data), len(dcm_files))

        # پردازش برای هر اندام
        for target_organ in target_organs:
            if target_organ not in Organ:
                print(f"⚠️ Organ '{target_organ}' not in Organ list. Skipping...")
                continue

            organ_index = Organ.index(target_organ)
            print(f"🧠 Processing organ: {target_organ} (index {organ_index})")

            for idx in range(num_slices):
                binary_mask = (label_data[idx] == organ_index).astype(np.uint8) * 255

                # 🔄 اعمال flip پایین به بالا
                binary_mask = np.flipud(binary_mask)

                # 🔄 چرخش ۹۰ درجه موافق عقربه‌های ساعت
                binary_mask = np.rot90(binary_mask, k=-1)

                dcm_path = dcm_files[idx]
                out_path = dcm_path.replace(DCM_data_dir, output_dir)
                out_path = out_path.replace(patient_id, f"{patient_id}/{target_organ}")
                out_path = out_path.replace('.dcm', '_OUT.png')
                out_path = out_path.replace('\\', '/')

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, binary_mask)

        print(f"✅ Finished patient: {patient_id}")

    print("\n🎉 All masks generated and saved!")
