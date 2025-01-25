# 3D-image-segmentation

The dataset for thia project is available on kaggle [MedicalDecathlon](https://medicaldecathlon.com/)
The dataset only contains 20 MRIs of the heart, and 10 images for testing. The objective of the challenge is to segment the left atrium from the MRI images.
The images are in the NIfTI format.

there are some files i created to successfully complete this and the description for them is given below:
-`main.py or main.ipynb`: A python script that will run the project.
- `analizedata.py or ipynb`: A file used to analyze the data
- `mydataset.py`: A custom torch dataset that will load the data.
- `mymodell.py`: A python script that will contain the model or models tested. 
- `training.py`: A python script that will train the model.

This project is part of our university course i.e, Data Science Meets Health Science
In analizedata.py
-The T2 image with the segmentation overlayed, displaying the middle slice of the dimension with the 
lowest resolution.

In mydataset.py
- Resized the images to same isospacing
- Cropped the images from the center to a predefined size (e.g. 256x256x256)
- Normalized the images to have zero mean and unit variance

In mymodell.py
We implemented the version of 3D-Unet
- 3D convolutional layers
- 3D max pooling layers
- 3D upsampling layers
- 3D batch normalization layers
- 3D residual blocks

In training.py
- Iterated by a number of epochs and batches, and evaluate the model on the validation set at the end of each epoch.
- Created a tensorboard writer and saved the following information for each epoch:
    - Training loss
    - Validation loss
    - A sample of the validation images (2) and the predicted segmentations. 
    - Additionally, save the model graph for the first epoch
- Kept track of the model with the lowest validation loss and save it to disk
