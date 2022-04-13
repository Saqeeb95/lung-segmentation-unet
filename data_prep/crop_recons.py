import os
import numpy as np
import nibabel as nib

def crop_recons(data_dir='./data/',
                bitmaskFilename='bitmask.nii.gz',
                bitmaskCroppedFilename='bitmask_cropped.nii.gz',
                reconFilename='nufft_recon.nii.gz',
                reconCroppedFilename='nufft_recon_cropped.nii.gz'):
    """Crops each image in the data directory with the specified filename to only include
    slices for which there are bitmasks in the corresponding bitmask file. Expects each 
    image to be in its own folder in the data directory, and saves the cropped image and 
    bitmask to the corresponding folder for each file."""
    for folder in os.listdir(data_dir):
        # Load Volumes
        bitmaskPath = data_dir + folder + '/' + bitmaskFilename
        bitmask = nib.load(bitmaskPath) 
        bitmask = np.array(bitmask.dataobj)    # Convert to numpy array

        reconPath = data_dir + folder + '/' + reconFilename
        recon = nib.load(reconPath) 
        recon = np.array(recon.dataobj)        # Convert to numpy array

        # Identify the first and last slices in the bitmask
        frontToBack = list(range(bitmask.shape[2]))
        backToFront = reversed(frontToBack)
        for i in frontToBack:
            slice = bitmask[:,:,i]
            if 1 in slice:
                firstSlice = i
                break
        for i in backToFront:
            slice = bitmask[:,:,i]
            if 1 in slice:
                lastSlice = i
                break
        print("Patient: {:2} First: {:2} Last: {:2} numSlices: {:2}".format(folder, firstSlice, lastSlice, lastSlice-firstSlice+1))

        # Crop both volumes to the desired slice range
        bitmaskCropped = bitmask[:,:,firstSlice:lastSlice+1]
        reconCropped = recon[:,:,firstSlice:lastSlice+1]
        
        # Convert back to nifti images and save
        bitmaskCropped = nib.Nifti1Image(bitmaskCropped, np.eye(4))
        bitmaskCroppedPath = data_dir + folder + '/' + bitmaskCroppedFilename
        nib.save(bitmaskCropped, bitmaskCroppedPath)

        reconCropped = nib.Nifti1Image(reconCropped, np.eye(4))
        reconCroppedPath = data_dir + folder + '/' + reconCroppedFilename
        nib.save(reconCropped, reconCroppedPath)