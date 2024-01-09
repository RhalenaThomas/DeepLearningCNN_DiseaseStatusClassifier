# Paths for preprocessed images

# Path for model
import argparse
from lime import lime_image
import os
import numpy as np
from tensorflow.keras.preprocessing import image

parser = argparse.ArgumentParser(description='Process input for test_model.')

parser.add_argument('model_name')
parser.add_argument('test_set_name')
args = parser.parse_args()

model_path = '/volume/models/' + args.model_name
test_path = '/volume/data/' + args.test_set_name + '/test_set'





def load_images(path, amount):
    # Put files into lists and return them as one list of size 4

    image_files = []
    for file in os.listdir(path):
        image_files.append(os.path.join(path,file))
        if len(image_files) > 4: 
            break
    return image_files


healthy = np.array(load_images(test_path + '/healthy'))
unhealthy = np.array(load_images(test_path + '/unhealthy'))
X_names = np.append(healthy, unhealthy)
X_healthy = []
X_unhealthy = []

for image in healthy:
    X_healthy.append(image.img_to_array(image.load_img(image, target_size=(64, 64))))

for image in unhealthy:
    X_unhealthy.append(image.img_to_array(image.load_img(image, target_size=(64, 64))))

X.append(image.img_to_array(image.load_img(healthy[0], target_size=(64, 64))))
X.append(image.img_to_array(image.load_img(healthy[1], target_size=(64, 64))))
X.append(image.img_to_array(image.load_img(unhealthy[7], target_size=(64, 64))))
X.append(image.img_to_array(image.load_img(unhealthy[8], target_size=(64, 64))))

X = np.array(X).astype('double')
# Load saved model
from tensorflow.keras.models import load_model

model = load_model('models/' + model_path)

preds = model.predict(X)
labels = ['healthy'] * len(healthy) + ['unhealthy'] * len(unhealthy)

explainer = lime_image.LimeImageExplainer(random_state=42)


import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

    explanation = explainer.explain_instance(X[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    image, mask  = explanation.get_image_and_mask(1, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0)

    plt.subplot(241)
    plt.grid(False)
    plt.imshow(mark_boundaries(X[0]/256, mask))
    plt.subplot(245)
    plt.imshow(X[0]/256)

    explanation = explainer.explain_instance(X[1], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    image, mask  = explanation.get_image_and_mask(1, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0)

    plt.subplot(242)
    plt.grid(False)
    plt.imshow(mark_boundaries(X[1]/256, mask))
    plt.subplot(246)
    plt.imshow(X[1]/256)

    explanation = explainer.explain_instance(X[2], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    image, mask  = explanation.get_image_and_mask(0, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0)

    plt.subplot(243)
    plt.grid(False)
    plt.imshow(mark_boundaries(X[2]/256, mask))
    plt.subplot(247)
    plt.imshow(X[2]/256)

    explanation = explainer.explain_instance(X[3], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    image, mask  = explanation.get_image_and_mask(0, positive_only=True, hide_rest=False, num_features=5, min_weight=0.0)

    plt.subplot(244)
    plt.grid(False)
    plt.imshow(mark_boundaries(X[3]/256, mask))
    plt.subplot(248)
    plt.imshow(X[3]/256)


    plt.show()