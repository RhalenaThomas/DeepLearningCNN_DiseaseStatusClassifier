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
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
train_gen = datagen.flow_from_directory(train_path, class_mode='categorical', batch_size=64, target_size=(64, 64), shuffle=True)
val_gen = datagen.flow_from_directory(train_path, class_mode='categorical', batch_size=64, target_size=(64, 64),  subset='validation')


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
