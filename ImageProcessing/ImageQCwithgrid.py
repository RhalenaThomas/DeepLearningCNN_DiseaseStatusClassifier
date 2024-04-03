from PIL import Image
import cv2
import numpy as np
import os
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
import matplotlib.pyplot as plt

### function for the grid crop and save assumes 3 channel images 

# Function to merge images

def merge_images(input_directory, output_directory=None):
    merged_images = []
    original_filenames = []  # List to store original filenames

    # Get a list of all image files in the input directory
    image_files = sorted([os.path.join(input_directory, file)
                          for file in os.listdir(input_directory) if "d0.TIF" in file])

    for img_path in image_files:
        try:
            # Load channels
            ch0 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            ch1 = cv2.imread(img_path.replace("d0", "d1"), cv2.IMREAD_GRAYSCALE)
            ch2 = cv2.imread(img_path.replace("d0", "d2"), cv2.IMREAD_GRAYSCALE)

            # Merge channels into an RGB image
            merged_image = np.zeros((ch0.shape[0], ch0.shape[1], 3), dtype=np.uint8)
            merged_image[:, :, 0] = ch0
            merged_image[:, :, 1] = ch1
            merged_image[:, :, 2] = ch2

            # Append merged image to list
            merged_images.append(merged_image)

            # Save merged image to output directory if provided
            if output_directory:
                base = os.path.basename(img_path)
                merged_image_path = os.path.join(output_directory, f"{os.path.splitext(base)[0]}.jpg")
                cv2.imwrite(merged_image_path, merged_image)

            # Append original filename to list
            original_filenames.append(os.path.basename(img_path))

        except Exception as e:
            print("Exception occurred with the following image:", img_path)
            print(repr(e))

    if output_directory is None:
        return merged_images, original_filenames

# Call merge_images function
original_image_path = "/Users/rhalenathomas/Desktop/temp_images/Test_images_QC/"
merged_images, original_filenames = merge_images(original_image_path)  

# Check if merge_images succeeded
if merged_images:
    # Define grid_crop_and_save function
    def grid_crop_and_save(merged_images, original_filenames, grid_images_folder):
        # Create output directory if it doesn't exist
        os.makedirs(grid_images_folder, exist_ok=True)

        for idx, (merged_image, filename) in enumerate(zip(merged_images, original_filenames)):
            for y in range(8):
                for x in range(8):
                    left = x * 138  # Adjusted for 1104x1104 images
                    upper = y * 138  # Adjusted for 1104x1104 images
                    right = left + 138  # Adjusted for 1104x1104 images
                    lower = upper + 138  # Adjusted for 1104x1104 images

                    # Crop the merged image to extract the current grid
                    grid_image = Image.fromarray(merged_image[left:right, upper:lower, :])

                    # Save the grid image with the original file name and grid position
                    grid_image_name = f"{filename.split('.')[0]}_grid_{x}_{y}.jpg"
                    grid_image_path = os.path.join(grid_images_folder, grid_image_name)
                    grid_image.save(grid_image_path)

    # Specify paths
    grid_images_folder_path = "/Users/rhalenathomas/Desktop/temp_images/Grid_images"

    # Perform grid cropping and save grid images
    grid_crop_and_save(merged_images, original_filenames, grid_images_folder_path)
else:
    print("No images were merged. Check the input directory.")


# Function to measure image features
def measure_features(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return None
    
    # Measure image features
    features = {}
    
    # Calculate image blur
    blur_score = cv2.Laplacian(image, cv2.CV_64F).var()
    features['blur_score'] = blur_score
    
    # Calculate intensity measurement (e.g., mean pixel value)
    intensity_value = np.mean(image)
    features['intensity_value'] = intensity_value
    
    # Thresholding to segment nuclei
    threshold_value = threshold_otsu(image)
    binary_image = image > threshold_value
    
    # Count nuclei
    labeled_image, num_features = cv2.connectedComponents(np.uint8(binary_image))
    num_nuclei = num_features
    features['num_nuclei'] = num_nuclei
    
    # Print image name and measures
    image_name = os.path.basename(image_path)
    print(f"Image Name: {image_name}")
    print(f"Blur Score: {blur_score}")
    print(f"Intensity Value: {intensity_value}")
    print(f"Number of Nuclei: {num_nuclei}")
    
    return features

# Function to remove low quality images based on thresholds
def remove_low_quality_images(image_folder, threshold_blur, threshold_nuclei, threshold_intensity, filtered_images_folder_path):
    filtered_images = []
    
    # Iterate over all image files in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        features = measure_features(image_path)
        if (features['blur_score'] < threshold_blur 
            and features['num_nuclei'] >= threshold_nuclei 
            and features['intensity_value'] < threshold_intensity):
            
            filtered_images.append(image_name)
            # Optionally, save the filtered image to a new folder
            filtered_image_path = os.path.join(filtered_images_folder_path, image_name)
            # Copy or move the filtered image to the output directory
            # For example:
            # shutil.copy(image_path, filtered_image_path)  # Copy the image
            # or
            # shutil.move(image_path, filtered_image_path)  # Move the image
            
            print(f"Image '{image_name}' passed all criteria and saved to '{filtered_images_folder_path}'")
    
    return filtered_images

# Function to remove low quality images based on thresholds
def remove_low_quality_images(image_folder, threshold_blur, threshold_nuclei, threshold_intensity, filtered_images_folder_path):
    filtered_images = []
    
    # Iterate over all image files in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        features = measure_features(image_path)
        if (features['blur_score'] < threshold_blur 
            and features['num_nuclei'] >= threshold_nuclei 
            and features['intensity_value'] < threshold_intensity):
            
            filtered_images.append(image_name)
            # Optionally, save the filtered image to a new folder
            filtered_image_path = os.path.join(filtered_images_folder_path, image_name)
            # Copy or move the filtered image to the output directory
            # For example:
            # shutil.copy(image_path, filtered_image_path)  # Copy the image
            # or
            # shutil.move(image_path, filtered_image_path)  # Move the image
            
            print(f"Image '{image_name}' passed all criteria and saved to '{filtered_images_folder_path}'")
    
    return filtered_images

# Filter thresholds
threshold_blur = 5  # Example threshold for blur score
threshold_nuclei = 1  # Example threshold for the minimum number of nuclei
threshold_intensity = 1  # Example threshold for intensity measurement



# Specify paths
original_image_path = "/Users/rhalenathomas/Desktop/temp_images/Test_images_QC/"
grid_images_folder_path = "/Users/rhalenathomas/Desktop/temp_images/Grid_images/"
filtered_images_folder_path = "/Users/rhalenathomas/Desktop/temp_images/FilteredImages/"

# Get a list of all grid image files
grid_image_files = [os.path.join(grid_images_folder_path, file) for file in os.listdir(grid_images_folder_path)]


# should save the filtered images to a the filteure_images_folder_path folder

filtered_images = remove_low_quality_images(grid_image_files, threshold_blur, threshold_nuclei, threshold_intensity, filtered_images_folder_path)


# Print the filtered images
print("Filtered images:", filtered_images)
