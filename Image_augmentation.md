#training with image augmentation

Each image in the training dataset will undergo random transformations according to the specified augmentation parameters.
These transformations include rotation, shifting, shearing, zooming, flipping, adjusting brightness and contrast, applying blur, enhancing edges, and adjusting saturation, among others.
Each image will have a random combination of these transformations applied to it, creating variations of the original image.
The original images will still be used as part of the training dataset, and they will be augmented with additional variations created by the random transformations.
During each training epoch, different augmented versions of the images will be generated, providing the model with a diverse set of examples to learn from and helping to improve its generalization ability.
By applying random augmentations to the training images, you effectively increase the variability of the training dataset, which can help the model learn to generalize better to unseen data and reduce overfitting. However, it's essential to strike a balance and avoid excessive augmentation, as it may lead to unrealistic images that could confuse the model.


```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define augmentation parameters
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,  # Rotate images randomly up to 20 degrees
    width_shift_range=0.2,  # Shift images horizontally by up to 20% of the width
    height_shift_range=0.2,  # Shift images vertically by up to 20% of the height
    shear_range=0.2,  # Shear intensity (shear angle in radians)
    zoom_range=0.2,  # Zoom range [1 - 0.2, 1 + 0.2]
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=True,  # Flip images vertically
    brightness_range=[0.5, 1.5],  # Adjust brightness between 0.5 and 1.5
    contrast_range=[0.5, 1.5],  # Adjust contrast between 0.5 and 1.5
    blur_range=[1, 3],  # Apply blur with kernel sizes between 1 and 3
    edge_enhance=True,  # Enhance edges of the images
    saturation_range=[0.5, 1.5]  # Adjust saturation between 0.5 and 1.5
)

# Only rescale validation data
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create data generators for training and validation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)


```