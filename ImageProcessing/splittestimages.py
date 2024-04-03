import os
import shutil
import random

def split_data(parent_dir, test_ratio=0.2):
    # Define the paths for the original healthy and unhealthy folders
    healthy_dir = os.path.join(parent_dir, 'healthy')
    unhealthy_dir = os.path.join(parent_dir, 'unhealthy')
    
    # Create the training and test folders
    training_dir = os.path.join(parent_dir, 'training')
    test_dir = os.path.join(parent_dir, 'test')
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subfolders for healthy and unhealthy in training and test folders
    training_healthy_dir = os.path.join(training_dir, 'healthy')
    training_unhealthy_dir = os.path.join(training_dir, 'unhealthy')
    test_healthy_dir = os.path.join(test_dir, 'healthy')
    test_unhealthy_dir = os.path.join(test_dir, 'unhealthy')
    os.makedirs(training_healthy_dir, exist_ok=True)
    os.makedirs(training_unhealthy_dir, exist_ok=True)
    os.makedirs(test_healthy_dir, exist_ok=True)
    os.makedirs(test_unhealthy_dir, exist_ok=True)
    
    # Get a list of all image files in the healthy and unhealthy directories
    healthy_images = os.listdir(healthy_dir)
    unhealthy_images = os.listdir(unhealthy_dir)
    
    # Shuffle the lists of images
    random.shuffle(healthy_images)
    random.shuffle(unhealthy_images)
    
    # Calculate the number of images for the test set based on the test_ratio
    num_test_healthy = int(len(healthy_images) * test_ratio)
    num_test_unhealthy = int(len(unhealthy_images) * test_ratio)
    
    # Move images from healthy and unhealthy directories to test directories
    for img in healthy_images[:num_test_healthy]:
        src = os.path.join(healthy_dir, img)
        dst = os.path.join(test_healthy_dir, img)
        shutil.copy(src, dst)
    for img in unhealthy_images[:num_test_unhealthy]:
        src = os.path.join(unhealthy_dir, img)
        dst = os.path.join(test_unhealthy_dir, img)
        shutil.copy(src, dst)
    
    # Move remaining images to training directories
    for img in healthy_images[num_test_healthy:]:
        src = os.path.join(healthy_dir, img)
        dst = os.path.join(training_healthy_dir, img)
        shutil.copy(src, dst)
    for img in unhealthy_images[num_test_unhealthy:]:
        src = os.path.join(unhealthy_dir, img)
        dst = os.path.join(training_unhealthy_dir, img)
        shutil.copy(src, dst)

# Specify the parent directory containing the 'healthy' and 'unhealthy' folders

# Example usage:
parent_directory = "/Users/rhalenathomas/Desktop/temp_images/A"

split_data(parent_directory)
print("Images are splilt")

