from PIL import Image
import numpy as np
import os

# Function to merge images
def merge_images(input_directory, output_directory=None):
    merged_images = []
    original_filenames = []  # Keep track of original file names
    
    # Get a list of all image files in the input directory
    image_files = os.listdir(input_directory)

    for filename in image_files:
        # Load the image
        img_path = os.path.join(input_directory, filename)
        img = Image.open(img_path)
        
        # Check the number of channels
        if img.mode == 'RGB':
            merged_image = np.array(img)
        else:
            # Convert grayscale or other formats to RGB
            merged_image = np.zeros((img.height, img.width, 3), dtype=np.uint8)
            merged_image[:, :, 0] = np.array(img)
            merged_image[:, :, 1] = np.array(img)
            merged_image[:, :, 2] = np.array(img)

        merged_images.append(merged_image)
        original_filenames.append(filename)  # Store original filename

        # Save merged image to output directory if provided
        if output_directory:
            prefix = filename.split("d")[0]  # Get everything before the "d" character
            merged_image_path = os.path.join(output_directory, f"{prefix}_merged.jpg")
            Image.fromarray(merged_image).save(merged_image_path)

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
