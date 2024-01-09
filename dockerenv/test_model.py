# test_model.py
#
# Used to test models and displays results (metrics / confusion matrix / roc curve)
#
# test_path - Name for data set of preprocessed images to test with
# model_name - Model to test (SavedModel format)
# weights_path - Load weights if needed as a path (Otherwise, None)
# plot_labels - Labels for confusion matrix [healthy, unhealthy]

import argparse

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sn
import matplotlib.pyplot as plt
from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


parser = argparse.ArgumentParser(description='Process input for test_model.')

parser.add_argument('model_name')
parser.add_argument('test_set_name')
parser.add_argument('--weights')
parser.add_argument('--lab', nargs=2, default = ['Control', 'Unhealthy'])

args = parser.parse_args()


#test_set_name = 'NPC_preprocessed_batch_3_plate_1'
#model_name = 'test_model'
#weights_path = None
#plot_labels = ['Control', 'Parkin KO']

# Create the data generator for the test set
model_path = '/volume/models/' + args.model_name
test_path = '/volume/data/' + args.test_set_name + '/test_set'
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
test_gen = datagen.flow_from_directory(test_path,
                                       class_mode='categorical',
                                       batch_size=64,
                                       target_size=(64, 64),
                                       shuffle=False,
                                       follow_links=True)
                                       

# Load saved model, and weights if necessary
model = load_model(model_path)

if args.weights:
    model.load_weights(args.weights)


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

plt.figure(figsize=(6, 5))
sn.set(font_scale=1.2)  # for label size
heatmap = sn.heatmap(cm,
                     annot=True,
                     annot_kws={"size": 12},
                     cbar=False,
                     cmap='Blues',
                     linewidths=2,
                     linecolor='black',
                     fmt='g',
                     xticklabels=["Pred. "+args.lab[0], "Pred. "+args.lab[1]],
                     yticklabels=["True "+args.lab[0], "True "+args.lab[1]])
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0)
heatmap.xaxis.tick_top()
heatmap.tick_params(length=0)
plt.gcf().subplots_adjust(left=0.25)
plt.savefig(model_path + '/test_' + args.test_set_name + '_cm.png')


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
plt.gcf().subplots_adjust(left=0.2)
plt.savefig(model_path + '/test_' + args.test_set_name + '_roc.png')

probabilities = [i[1] for i in predictions]
plt.clf()
plt.title('Probability Distribution')
plt.hist(probabilities, bins=20)
plt.xlim([0, 1])
plt.ylabel('# of Predictions')
plt.xlabel('Probability')
plt.savefig(model_path + '/test_' + args.test_set_name + '_hist.png')

np.savetxt(model_path + '/test_' + args.test_set_name + '_predictions.txt', predictions, delimiter=",")
np.savetxt(model_path + '/test_' + args.test_set_name + '_true_classes.txt', true_classes, delimiter=",")
np.savetxt(model_path + '/test_' + args.test_set_name + '_filepaths.txt', np.array(test_gen.filepaths), delimiter=",", fmt='%s')
predictions = np.array(predictions)

predictions_1 = predictions[true_classes == 1]
predictions_0 = predictions[true_classes == 0]

best_1_image = img_to_array(load_img(test_gen.filepaths[np.argmax(predictions_1[:,1])], target_size=(64, 64))).astype("double")
worst_1_image = img_to_array(load_img(test_gen.filepaths[np.argmin(predictions_1[:,1])], target_size=(64, 64))).astype("double")
best_0_image = img_to_array(load_img(test_gen.filepaths[np.argmax(predictions_0[:,0])], target_size=(64, 64))).astype("double")
worst_0_image = img_to_array(load_img(test_gen.filepaths[np.argmin(predictions_0[:,0])], target_size=(64, 64))).astype("double")






explainer = lime_image.LimeImageExplainer(random_state=42)


explanation = explainer.explain_instance(best_0_image, model.predict, top_labels=5, hide_color=0, num_samples=1000)
image, mask  = explanation.get_image_and_mask(1, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0)

ax = plt.subplot(241)
plt.grid(False)
plt.axis('off')
ax.set_title("C/C")
plt.imshow(mark_boundaries(best_0_image/256, mask), interpolation='nearest')
plt.subplot(245)
plt.grid(False)
plt.axis('off')
plt.imshow(best_0_image/256, interpolation='nearest')

explanation = explainer.explain_instance(worst_0_image, model.predict, top_labels=5, hide_color=0, num_samples=1000)
image, mask  = explanation.get_image_and_mask(1, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0)

ax = plt.subplot(242)
plt.grid(False)
plt.axis('off')
ax.set_title("KO/C")
plt.imshow(mark_boundaries(worst_0_image/256, mask), interpolation='nearest')
plt.subplot(246)
plt.grid(False)
plt.axis('off')
plt.imshow(worst_0_image/256, interpolation='nearest')

explanation = explainer.explain_instance(best_1_image, model.predict, top_labels=5, hide_color=0, num_samples=1000)
image, mask  = explanation.get_image_and_mask(0, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0)

ax = plt.subplot(243)
plt.grid(False)
plt.axis('off')
ax.set_title("KO/KO")
plt.imshow(mark_boundaries(best_1_image/256, mask), interpolation='nearest')
plt.subplot(247)
plt.grid(False)
plt.axis('off')
plt.imshow(best_1_image/256, interpolation='nearest')

explanation = explainer.explain_instance(worst_1_image, model.predict, top_labels=5, hide_color=0, num_samples=1000)
image, mask  = explanation.get_image_and_mask(0, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0)

ax = plt.subplot(244)
plt.grid(False)
plt.axis('off')
ax.set_title("C/KO")
plt.imshow(mark_boundaries(worst_1_image/256, mask), interpolation='nearest')
plt.subplot(248)
plt.grid(False)
plt.axis('off')
plt.imshow(worst_1_image/256, interpolation='nearest')

plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(model_path + '/test_' + args.test_set_name + '_lime.png', bbox_inches='tight')