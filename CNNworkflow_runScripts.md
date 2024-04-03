# CNN workflow

1. Image preprocessing
    a) downsample
    b) grid
    c) cell segmentation
    d) QC of images
    e) splitting data to separate completely hidden test data - then this data can be tested in other models. 
2. Build model and save starting model.
    a) binary categorical
    b) multi category categorical
    c) model with randomized activation functions
3. Train model 
    a) standard
    b) image augmentation
    c) internal cross-validation
4. Test model with the hidden data
    a) Matching the model
    b) other data to test generalization


# Running
1. Edit all scripts and save with a different file name for each experiment - the file pathways will be changed inside the scripts.
2. Establish the virtual environment.
3. Call each python script. 

# Virtual enviroment
See main readme for other instructions

```
source Script/activate

```

# Step 1 - Image preprocessing
Choose the processing script for the method you will use.
Navigate to the location of your script that you have edited.
Run in terminal / command line 
Current models all take in 64x64 pixle images.

To compare channel contributions it is at this stage that these would be included or excluded.
A different script is needed to produce the different channel processing.
Here we assume the high content images have matching image names with designations for the different channels.

```
# Image merging - create a multi channel in this case RBG image and save as a jpg


```

Split processed images into training and test batches.
Note that training and validation splits will occur inside the model.
We opt to split the data before the grid crop or cell seqmentation.  This way sections of one image cannot be present in training and test data.

```
# Split 80 % training and 20 % test data
python training_split.py

```

Futher process images selecting the method desired. 
The training and test data must be processed the same way but separaetely

```
# Cuts each image in 8X8 grid creating 64 images that are 138x138 px
# These images are the measured for intensity, blurr, and number of nuclei - extreme images or those without cells are removed
# the images are then down sampled to 64x64 px to be entered into the model.

python processGridQC.py

```


# Step 2 - make the original model
The number of categories in the ouptput needs to be define here. 
This creates the model architecture with random weights to be put into the training script.

```
# the model will recieve only the training set which will be split into training and validation
# the base script doesn't run cross validation

python create_model_original.py

```

# Step 3 - train the model
Here the model will be trained and statistics on the model for training will be output.
The model expects an image folder for each category.

```
python train_model.py

```

# Step 4 - test the hidden test data
Here the hidden test data will be predicted by the model
The model expects an image folder for each category.

```
python test_model.py

```

