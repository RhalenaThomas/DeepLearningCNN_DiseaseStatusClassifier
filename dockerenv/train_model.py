# train_model.py
#
# Used to train models with data generator
#
# train_set_name - Name for data set of preprocessed images
# model_start_name - Name of model to begin training with 'model_original' (SavedModel format)
# model_name - Name to save newly trained model

import argparse

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser(description='Process input for train_model.')

parser.add_argument('model_start_name')
parser.add_argument('train_set_names')
parser.add_argument('val_set_names')
parser.add_argument('model_name')
parser.add_argument('--steps', default=64, type=int)
parser.add_argument('--epoch', default=25, type=int)
parser.add_argument('--pat', default=50, type=int)

args = parser.parse_args()


#model_start_name = 'model_original'
#train_set_name = 'NPC_preprocessed_batch_3_plate_1'
#model_name = 'test_model'
#steps_per_epoch = 64
#max_epochs = 1
#early_stopping_patience = 10


###

train_paths = ['/volume/data/' + name + '/train_set' for name in args.train_set_names.split(',')]
val_paths = ['/volume/data/' + name + '/train_set' for name in args.val_set_names.split(',')]
model_start_path = '/volume/models/' + args.model_start_name
model_path = '/volume/models/' + args.model_name

steps_per_epoch = args.steps
max_epochs = args.epoch
early_stopping_patience = args.pat

train_gens = []
val_gens = []

# Create data generators
for train_path in train_paths:
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2, featurewise_center=True, featurewise_std_normalization=True)
    train_gen = datagen.flow_from_directory(train_path, class_mode='categorical', batch_size=64, target_size=(64, 64), shuffle=True, subset='training')
    train_gens.append(train_gen)

for val_path in val_paths:
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2, featurewise_center=True, featurewise_std_normalization=True)
    val_gen = datagen.flow_from_directory(train_path, class_mode='categorical', batch_size=64, target_size=(64, 64), subset='validation')
    val_gens.append(val_gen)

class CombinedGen():
    def __init__(self, gens):
        self.gens = gens
        self.shape = (sum([len(g) for g in self.gens]), 0)

    def generate(self):
        while True:
            for g in self.gens:
                yield next(g)

    def __len__(self):
        return sum([len(g) for g in self.gens])

# Load saved model and display the model's architecture
model = load_model(model_start_path)
print(model.summary())



# Pregenerate folder for end model
model.save(model_path)

def saveListToFile(listname, pathtosave):
    file1 = open(pathtosave,"w") 
    for i in listname:
        file1.writelines("{}\n".format(i))    
    file1.close() 

saveListToFile([steps_per_epoch, max_epochs, early_stopping_patience, train_paths, val_paths], model_path + "parameters.txt")

# Set callback functions to early stop training and save the best model so far
#callbacks = [EarlyStopping(monitor='val_loss', patience=early_stopping_patience),
#             ModelCheckpoint(filepath= model_path + '/best_weights.h5', monitor='val_loss', save_best_only=True)]


#K.set_value(model.optimizer.learning_rate, 0.001)


cg = CombinedGen(train_gens)
cgv = CombinedGen(val_gens)
# Model Training
history = model.fit_generator(
        cg.generate(),
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
#        callbacks=callbacks,  # Early stopping
        validation_data=cgv.generate(),
        validation_steps=steps_per_epoch,
        shuffle=True
        )

print("fit complete")

# Save the current model after training
model.save(model_path)

# List all data in history
print(history.history.keys())

saveListToFile([history], model_path + "history.txt")


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



