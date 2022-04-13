import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class NiftiDataset2D(Dataset):
    def __init__(self, reconPath, bitmaskPath, normalize=False, transform=None, bitmask_tfms=None, rotate=None, includeFilename = False):
        # Load all images into a list given folder paths for the recons and bitmasks
        self.reconList = [nib.load(reconPath+file) for file in os.listdir(reconPath)]
        self.bitmaskList = [nib.load(bitmaskPath+file) for file in os.listdir(bitmaskPath)]
        self.transform = transform
        self.bitmask_tfms = bitmask_tfms
        self.normalize = normalize
        self.rotate = rotate
        self.includeFilename = includeFilename
        if self.includeFilename:
            self.reconFiles = [file for file in os.listdir(reconPath)]

    def __len__(self):
        return len(self.reconList)

    def __getitem__(self, idx):
        # Convert to numpy array and then tensor
        recon = np.array(self.reconList[idx].dataobj).astype('float32')
        
        # Normalize recons to values between [0,1]
        if self.normalize:
            minVal = np.min(recon)
            maxVal = np.max(recon)
            recon = (recon-minVal)/(maxVal-minVal)
        recon = torch.from_numpy(recon)

        # Unsqueeze so images have a dimension for channels
        recon = recon.unsqueeze(0)

        # Apply transforms
        if self.transform:
            recon = self.transform(recon)

        if self.rotate:
            numRotations = self.rotate//90
            recon = torch.rot90(recon, numRotations, [1,2])

        # Bitmask needs to have num_channels = num_classes
        # With two classes, each channel needs to have values 1 - other channel
        bitmask = np.array(self.bitmaskList[idx].dataobj).astype('float32')   # Convert from nifti to numpy array
        bitmask = np.repeat(bitmask[np.newaxis, :, :], 2, axis = 0) # Repeat array into channel dimension
        bitmask[1,:,:] = 1 - bitmask[0,:,:]                 # bitmask logits need to be opposite values in opposite channels
        bitmask = torch.from_numpy(bitmask)
        
        if self.bitmask_tfms:
            bitmask = self.bitmask_tfms(bitmask)

        if self.rotate:
            numRotations = self.rotate // 90
            bitmask = torch.rot90(bitmask, numRotations, [1,2])
        
        if self.includeFilename:
            filename = self.reconFiles[idx]
            return recon, bitmask, filename
        return recon, bitmask