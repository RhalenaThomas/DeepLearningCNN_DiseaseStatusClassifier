# test_model.py
#
# Used to test models and displays results (metrics / confusion matrix / roc curve)
#
# test_path - Paths for preprocessed images
# model_path - Model to test as a path (SavedModel format)
# weights_path - Load weights if needed as a path (Otherwise, None)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sn
import matplotlib.pyplot as plt

test_path = 'data/NPC_3/test_set'
model_path = 'models/model_NPC_3_more'
weights_path = None


# Create the data generator for the test set
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
test_gen = datagen.flow_from_directory(test_path,
                                       class_mode='categorical',
                                       batch_size=64,
                                       target_size=(64, 64),
                                       shuffle=False)

# Load saved model, and weights if necessary
model = load_model(model_path)

if weights_path:
    model.load_weights(weights_path)


# Generate model predictions and true classes
#
# predictions - list of probabilities of predicting each class.
# predicted_classes - prediction for each image based on probabilities
predictions = model.predict_generator(test_gen)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())


# Print the classification report and overall accuracy
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
_, acc = model.evaluate_generator(test_gen, steps=len(test_gen), verbose=0)
print('Model Accuracy: %.3f' % (acc * 100.0))


# Generate and plot the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(7, 7))
sn.set(font_scale=1.2)  # for label size
heatmap = sn.heatmap(cm,
                     annot=True,
                     annot_kws={"size": 12},
                     cbar=False,
                     cmap='Greens',
                     linewidths=1,
                     linecolor='black',
                     fmt='g')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.invert_yaxis()
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=90)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Generate and plot the Roc Curve
fpr, tpr, threshold = roc_curve(true_classes, predicted_classes)
roc_auc = auc(fpr, tpr)

plt.clf()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
