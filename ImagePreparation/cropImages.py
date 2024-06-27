import random
from PIL import Image
import os

def crop(images_dir):

    images = os.listdir(images_dir)
    image_size = 1024
    crop_size = 256

    for img in images:
        try:

            image_path = os.path.join(images_dir, img)
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

        except Exception as e:
            print()
            print(f"There was wn exception during cropping with the following file: {img}")
            print(repr(e))




    print("Successfully finished cropping images")
