import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt

def contours_to_bitmask(data_dir='./data/', regFilename='nufft_segmentation.nii.gz', contours=[0,1]):
    """Converts a registration file containing multiple classes of contours into a registration containing
    only the contours specified by the `contours` array. Expects each registration image to be in its own
    folder in the data directory, and saves the new registration to the corresponding folder for each file."""
    for folder in os.listdir(data_dir):
        # Load Image
        regPath = data_dir + folder + '/' + regFilename
        reg = nib.load(regPath)         # Registration image
        reg = np.array(reg.dataobj)     # Convert to numpy array
        print(reg.shape)                # Sanity check

        # Get rid of the non-lung contours
        reg = np.where(reg not in contours, 0, reg)
        # Try this line if the above line doesn't work, set the condition as you need
        # reg = np.where(reg > 1, 0, reg)

        # Creat bitmask
        bitmask = np.zeros(reg.shape)
        for i in range(reg.shape[2]):   # Iterate over coronal slices
            bitmask[:, :, i] = ndimage.binary_fill_holes(reg[:, :, i])

        # Convert back to nifti image and save
        bitmask = nib.Nifti1Image(bitmask, np.eye(4))
        bitmaskPath = data_dir + folder + '/' + 'bitmask.nii.gz'
        nib.save(bitmask, bitmaskPath)
