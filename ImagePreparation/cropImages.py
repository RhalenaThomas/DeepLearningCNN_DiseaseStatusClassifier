import random
from PIL import Image
import os

def crop(image_path):

    image_size = 1104
    crop_size = 256

    image_path = os.path.join(image_folder, image_path)
    image = Image.open(image_path)

    n_crops = image_size // crop_size

    offset_max = (image_size - crop_size * n_crops) // n_crops

    # Iterate through the grid and crop the image
    cropped_images = []
    for i in range(n_crops):
        for j in range(n_crops):
            offset = random.randint(0, offset_max)
            upper = i * crop_size + offset
            left = j * crop_size + offset
            right = left + crop_size
            lower = upper + crop_size 
            crop_box = (left, upper, right, lower)
        
            cropped_image = image.crop(crop_box)
            cropped_images.append(cropped_image)

            # Save the cropped image
            cropped_image_path = f"{image_path[:-4]}_{i}_{j}.TIF"
            cropped_image.save(cropped_image_path)

    os.remove(image_path)

    #print(f"Successfully cropped and saved {len(cropped_images)} images.")


image_folder = "some folder"

images = os.listdir(image_folder)

for img in images:
     crop(img)


print("Successfully finished cropping images")







