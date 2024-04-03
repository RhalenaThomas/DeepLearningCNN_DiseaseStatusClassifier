import os
import cv2
import numpy as np
import pandas as pd

def detect_nuclei(binary_image):
    # Label connected components to detect nuclei
    labeled_image, num_nuclei = cv2.connectedComponents(np.uint8(binary_image))
    return num_nuclei

def detect_bright_blobs(image):
    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to detect bright areas
    _, binary_image = cv2.threshold(grayscale_image, 200, 255, cv2.THRESH_BINARY)
    
    # Invert binary image
    binary_image = cv2.bitwise_not(binary_image)
    
    # Detect and count bright blobs
    _, num_bright_blobs = cv2.connectedComponents(binary_image)
    return num_bright_blobs

def calculate_mean_intensity(image):
    # Split image into channels
    b, g, r = cv2.split(image)
    
    # Calculate mean intensity for each channel
    mean_intensity_ch1 = np.mean(b)
    mean_intensity_ch2 = np.mean(g)
    mean_intensity_ch3 = np.mean(r)
    return mean_intensity_ch1, mean_intensity_ch2, mean_intensity_ch3

def process_images(input_folder):
    # Initialize lists to store data
    image_filenames = []
    nuclei_counts = []
    bright_blob_counts = []
    mean_intensity_ch1_list = []
    mean_intensity_ch2_list = []
    mean_intensity_ch3_list = []
    
    # Process each image in the input folder
    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)
        
        # Load image
        image = cv2.imread(image_path)
        
        # Check if image loaded successfully
        if image is None:
            print(f"Error: Unable to load image '{image_path}'")
            continue
        
        # Split image into channels
        ch1, _, _ = cv2.split(image)
        
        # Threshold channel 1 to segment nuclei
        _, binary_image = cv2.threshold(ch1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Detect and count nuclei
        num_nuclei = detect_nuclei(binary_image)
        
        # Detect and count bright blobs
        num_bright_blobs = detect_bright_blobs(image)
        
        # Calculate mean intensity for each channel
        mean_intensity_ch1, mean_intensity_ch2, mean_intensity_ch3 = calculate_mean_intensity(image)
        
        # Append data to lists
        image_filenames.append(image_filename)
        nuclei_counts.append(num_nuclei)
        bright_blob_counts.append(num_bright_blobs)
        mean_intensity_ch1_list.append(mean_intensity_ch1)
        mean_intensity_ch2_list.append(mean_intensity_ch2)
        mean_intensity_ch3_list.append(mean_intensity_ch3)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ImageFilename': image_filenames,
        'NucleiCount': nuclei_counts,
        'BrightBlobCount': bright_blob_counts,
        'MeanIntensityCh1': mean_intensity_ch1_list,
        'MeanIntensityCh2': mean_intensity_ch2_list,
        'MeanIntensityCh3': mean_intensity_ch3_list
    })
    
    # Save DataFrame to CSV
    output_csv_path = os.path.join(input_folder, 'image_features.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file saved to: {output_csv_path}")

# Specify input folder containing images
input_folder = "/Users/rhalenathomas/Desktop/temp_images/Grid_images/"

# Process images and save features to CSV
process_images(input_folder)
