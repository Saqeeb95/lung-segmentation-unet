# Overview

This U-Net model is used for 2D segmentation of lungs in MRI chest images (coronal views). The model itself was forked from [this github repo](https://github.com/milesial/Pytorch-UNet) and modified to work with the dataset and my custom dataloader and training loop. The dataset this was trained on was 43 3D MRI volumes of a mix of healthy volunteers and COVID-recovered patients. Lung, heart and liver registration was done slice by slice in coronal views by another member of my lab. Registrations were not applied to all slices containing lungs, making 3D lung segmentation difficult. I plan to segment the remaining slices manually before attempting to train a 3D segmentation model. This will allow better estimations of tidal volume, which a goal of this work.

## Installation

Clone the repo and run through the cells of the train-model.ipynb notebook. If you are missing any dependecies, try:

```bash
pip install -r requirements.txt
```



## Usage

### Data Prep

The `data_prep` folder contains several python scripts which can be imported to carry out a number of data preprocessing steps. If working with the xCovid study images (or any images that start out with 3D registration contours rather than 2D bitmasks), each patient's image and contour registration files should be in its own folder inside a data folder. To convert them into 2D slices ready for training, import and run the three files/functions below in order.

`contours_to_bitmask.py` - Converts a registration file containing multiple classes of contours into a registration containing only the contours specified by the `contours` array. Expects each registration image to be in its own folder in the data directory, and saves the new registration to the folder where it found the file.

`crop_recons.py` - Crops each image in the data directory with the specified filename (default=`nufft_recon.nii.gz`) to only include slices for which there are bitmasks in the corresponding bitmask file. Expects each image to be in its own folder in the data directory, and saves the cropped image and bitmask to the corresponding folder for each file with the specified target filename (default = `nufft_recon_cropped.nii.gz`).

`make_2d_dataset.py` - Grabs each image and bitmask from their respective folders in the data directory and saves each slice of each image directly into the specified image(recon) and bitmask paths as its own file. All recon (image) slices/files will be saved to the recon destination path and all bitmask (ground truth) slices/files will be saved to the bitmask destination path.

### Training the Data

Once the data is prepared (each slice has its own image and bitmask file) and all python requirements are met, you may run the `train-model.ipynb` notebook to train the model. This notebook will import all necessary modules. It uses a custom pytorch dataset class (`NiftiDataSet2D` in `niftiloader.py`) which enables the loading of NIFTI images. This dataset class expects all 2D slices to have a file in the recon folder for the image and another file for the bitmask in the bitmask folder.

If you want to make the model deterministic and reproducible, uncomment the last line in the second cell (`set_seed(num)`) and run that cell. 

In the third cell, specify the paths for the images (`reconPath`) and bitmasks (`bitmaskPath`). The `image_size` variable is used to centre crop all images to the same size. If the anatomy you want to segment is not in the centre of the images, you'll have to change this to something else.

In the fourth cell, the dataset is split into training and validation sets with the ratio set by `valid_percent`. These sets are used to get the mean and standard deviation across the data, so that the images can be normalized to have mean = 0 and std = 1.

In the sixth cell, another instance of the dataset is split into training and validation sets once again, this time with the tt.Normalize transform which normalizes the mean and std according to the stats calculated in the previous two cells. This cell also allows you to specify which transformations you want to perform on both the training and validation data. Currently both sets just have normalization and the same image crop as in the third cell.

The training occurs in the 10th cell of the notebook. Currently it's set to load a pretrained model from a file but if you don't have that, change `trainModel` to `True` before running. You can also change the hyperparameters in the same cell.

The cells below allow for visualization of some segmentations on the validation set along with Dice and ASSD scores. Hyperparameter tuning is available through the use of the `Optuna` module.

#### Optuna

The training notebook uses `Optuna` to run a hyperparameter search. As of now it searches for:

`lr` - The learning rate. Picks a number between 1e-6 and 1e-2, with log uniform probability.  

`optimizer` - Which optimizer to use. It chooses between SGD, Adam, and RMSprop.

`weight_decay` - Chooses a value for weight decay (regularization). It picks from a log uniform distribution between 1e-10 and 1e-1. Refer to the [pytorch documentation](https://pytorch.org/docs/stable/optim.html) to learn more.

`lr_scheduler` - The learning rate scheduler. Currently it picks between `ReduceLROnPlateau`, `OneCycleLR`, and `LambdaLR`. Refer to the [pytorch documentation]() to learn more.  
Note: the `LambdaLR` optimizer takes in a python lambda function to calculate the multiplier that scales the learning rate. The current lambda function is `lambda1 = lambda lr: 1` meaning it always returns 1. I set it this way so that the learning rate would stay the same the entire time.  

`loss_function` - The loss function. Chooses between cross entropy, dice loss, or both added together. Unlike the other parameters which are decided and initialized beforehand and passed to the training loop function, the loss function is simply passed to the training loop function as a string, and the loss calculation is handled inside function.

If you want to change any of the parameters, do so in the third-from-last cell in the notebook. The parameters are specified in a `config` dictionary inside the `optunaObjective` function definition. You'll also have to change the if statements for some of the hyperparameters. If you change the loss function choices, you'll have to change the loss function calculation in the training loop as well (inside `train.py`)

## License
[MIT](https://choosealicense.com/licenses/mit/)