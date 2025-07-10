import nibabel as nib
import pydicom
from pydicom.dataset import FileDataset
import numpy as np
import os
import datetime

def nii_to_dicom(nii_path, output_folder):
    # Load the NIfTI image
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    affine = nii_img.affine
    num_slices = data.shape[2]

    # Extract base name of NIfTI file (without extension)
    nii_base = os.path.splitext(os.path.basename(nii_path))[0]
    nii_base = nii_base.split('.')[0]
    # Create a subfolder named after the NIfTI file
    dicom_subfolder = os.path.join(output_folder, nii_base)
    os.makedirs(dicom_subfolder, exist_ok=True)

    for i in range(num_slices):
        filename = os.path.join(dicom_subfolder, f'slice_{i:03d}.dcm')

        # Create File Meta Information
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9.0"
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        # Create the FileDataset instance
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Fill in DICOM tags
        dt = datetime.datetime.now()
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.Modality = "MR"
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.StudyDate = dt.strftime('%Y%m%d')
        ds.StudyTime = dt.strftime('%H%M%S')

        ds.Rows, ds.Columns = data.shape[:2]
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [float(affine[0,3]), float(affine[1,3]), float(affine[2,3] + i)]
        ds.ImageOrientationPatient = [1,0,0,0,1,0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1

        # Ensure pixel data is uint16
        pixel_array = data[:, :, i].astype(np.uint16)
        ds.PixelData = pixel_array.tobytes()

        # Save DICOM file
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(filename)

    print(f"✅ DICOM series saved to: {dicom_subfolder}")

