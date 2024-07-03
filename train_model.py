# train_model.py
#
# Used to train models with data generator
#
# train_path - Paths for preprocessed images
# model_start - Path for model to begin training with 'model_original' (SavedModel format)
# model_name - Path to save newly trained model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

train_set_name = 'XCL_NPC_2_3'
model_start_name = 'model_23052024'
model_name = 'model_23052024_trained'
steps_per_epoch = 64
max_epochs = 5000
early_stopping_patience = 200


###

train_path = '/export02/data/CNN_deepLearning/HealthyUnhealthyClassifier_Clean/data_models/data/AIW-ParkinKO'
model_start_path = './models/' + model_start_name
model_path = './models/' + model_name

# Create data generators
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

train_gen = train_datagen.flow_from_directory(train_path, class_mode='categorical', batch_size=64, target_size=(64, 64), shuffle=True)
val_gen = val_datagen.flow_from_directory(train_path, class_mode='categorical', batch_size=64, target_size=(64, 64),  subset='validation')


# Load saved model and display the model's architecture
model = load_model(model_start_path)
print(model.summary())


# Pregenerate folder for end model
model.save(model_path)


# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=early_stopping_patience),
             ModelCheckpoint(filepath='models/' + model_path + '/best_weights.h5', monitor='val_loss', save_best_only=True)]


# Model Training
history = model.fit_generator(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        callbacks=callbacks,  # Early stopping
        validation_data=val_gen,
        shuffle=True
        )

# Save the current model after training
model.save(model_path)

# List all data in history
print(history.history.keys())

# Create plots for accuracy and loss during training and save in model folder
# Plot accuracy history and save it into model directory
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(model_path + '/training_accuracy.png')
plt.clf()

# Plot loss history and save it into model directory
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(model_path + '/training_loss.png')
