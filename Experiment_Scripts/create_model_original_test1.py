# create_model_original.py
#
# Used to create the original model architecture

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def create_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = create_model()
print(model.summary())

# Save the entire model as a SavedModel.
model.save('/Users/rhalenathomas/GITHUB/DeepLearningCNN_DiseaseStatusClassifier/models/model_original2')


