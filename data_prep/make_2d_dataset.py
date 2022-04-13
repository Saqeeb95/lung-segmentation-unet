import os
import numpy as np
import nibabel as nib

def make_2d_dataset(data_dir='./data/',
                    bitmaskCroppedFilename='bitmask_cropped.nii.gz',
                    bitmaskDestination='./2d/bitmask/',
                    reconCroppedFilename='nufft_recon_cropped.nii.gz',
                    reconDestination = './2d/recon/',
                    rotate = None):
    """Grabs each image and bitmask from their respective folders in the data directory and 
    saves each slice of each image directly into the specified image(recon) and bitmask paths"""
    for folder in os.listdir(data_dir):
        # Load Volumes
        bitmaskCroppedPath = data_dir + folder + '/' + bitmaskCroppedFilename
        bitmask = nib.load(bitmaskCroppedPath) 
        bitmask = np.array(bitmask.dataobj)    # Convert to numpy array

        reconCroppedPath = data_dir + folder + '/' + reconCroppedFilename
        recon = nib.load(reconCroppedPath) 
        recon = np.array(recon.dataobj)        # Convert to numpy array

        if rotate:
            numRotations = rotate // 90        # Rotations happen counterclockwise
            for _ in range(numRotations):
                bitmask = np.rot90(bitmask)
                recon = np.rot90(recon)

        for slice in range(bitmask.shape[2]):
            # Split up bitmask slices and save as nifti files
            bitmaskSlice = bitmask[:,:,slice]
            bitmaskSlice = nib.Nifti1Image(bitmaskSlice, np.eye(4))
            bitmaskSlicePath = bitmaskDestination + 'bitmaskSlice-' + folder + '-' + str(slice) + '.nii.gz'
            nib.save(bitmaskSlice, bitmaskSlicePath)

            # Split up recon slices and save as nifti files
            reconSlice = recon[:,:,slice]
            reconSlice = nib.Nifti1Image(reconSlice, np.eye(4))
            reconSlicePath = reconDestination + 'reconSlice-' + folder + '-' + str(slice) + '.nii.gz'
            nib.save(reconSlice, reconSlicePath)