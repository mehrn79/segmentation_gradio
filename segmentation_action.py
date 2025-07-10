from nifti2dicom import nii_to_dicom
from segment import segmentor
from generate_mask import generate_mask

def action(nii_url,target_organs):

   nii_to_dicom(nii_url, "Images_dicom") 
   segmentor()
   generate_mask(target_organs)



