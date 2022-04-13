import json
import torch
from torch.utils.data import DataLoader


def mean_std(batch):
    '''Calculate the means and standard deviations of all images in a batch by colour channel'''
    images, labels = next(iter(batch))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    return mean, std

def denormalize(images, means, stds, num_channels=1):
    means = torch.tensor(means).reshape(1, num_channels, 1, 1)
    stds = torch.tensor(stds).reshape(1, num_channels, 1, 1)
    return images * stds + means

def clamp0To1(images):
    def clampImage(image):
        maxVal = torch.max(image)
        minVal = torch.min(image)
        return (image-minVal) / (maxVal-minVal)
    if images.dim == 2:
        return clampImage(images)
    for i in range(images.size()[0]):
        images[i,0,:,:] = clampImage(images[i,0,:,:])
    return images

def get_mean_and_std(train_ds, data_prep_path):
    try:
        with open(data_prep_path+'stats.data', 'r') as filehandle: # open file for reading
            stats = json.load(filehandle)
        print("Loaded mean and standard deviation from file.")
        print(stats)
        mean, std = stats
        write = False
    except:
        print("stats.data does not exist in the", data_prep_path, "directory. Calculating the mean and standard deviation:")
        write = True

    if write:
        # Create a temporary dataset to get the mean and std
        dl = DataLoader(train_ds, batch_size=len(train_ds))

        mean, std = mean_std(dl)
        stats = mean.tolist(), std.tolist()
        print(f'Mean: {mean.item()}, StD: {std.item()}')

        with open(data_prep_path+'stats.data', 'w') as filehandle: # open file for writing
            json.dump((stats), filehandle)
        print("Stats saved in stats.data.")
    return mean, std