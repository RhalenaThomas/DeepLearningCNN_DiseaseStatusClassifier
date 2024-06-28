import os
import shutil
import random
import csv

def organize(images_folder, parent_dir, healthy_columns, unhealthy_columns):

    '''
    Given a set of images, class them into healthy and unhealthy folder depending on their column
    return a list of healthy triples (row, column, field) and triple of unhealthy
    '''

    healthy_wells = []
    unhealthy_wells = []

    healthy_dir = os.path.join(parent_dir, 'healthy')
    unhealthy_dir = os.path.join(parent_dir, 'unhealthy')

    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(unhealthy_dir, exist_ok=True)

    images_files = os.listdir(images_folder)



    for file in images_files:

        if "TIF" in file:
            filename_split = file.split("_")
            well = filename_split[2]
            col = well[1:3]
            row = well[0]
            
            if col in healthy_columns:
                image_path = os.path.join(images_folder, file)
                shutil.copy(image_path, healthy_dir)
                healthy_wells.append(well)
            elif col in unhealthy_columns :
                image_path = os.path.join(images_folder, file)
                shutil.copy(image_path, unhealthy_dir)
                unhealthy_wells.append(well)



    print("Images are organized into healthy and unhealthy directories.")
    print(healthy_wells)
    print(unhealthy_wells)
    return healthy_wells, unhealthy_wells



def split(parent_dir, healthy_wells, unhealthy_wells, index_file, prefix, test_ratio=0.2):

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
    # healthy_images = os.listdir(healthy_dir)
    # unhealthy_images = os.listdir(unhealthy_dir)


    # Shuffle the lists of healthy and unhealthy triples
    random.shuffle(healthy_wells)
    random.shuffle(unhealthy_wells)



    # Calculate the number of images for the test set based on the test_ratio
    # Divide by 3 since all the three channels should be in the same folder
    num_test_healthy = int(len(healthy_wells) * test_ratio) // 3
    num_test_unhealthy = int(len(unhealthy_wells) * test_ratio) // 3

    splitted_images = []

    # Move images from healthy and unhealthy directories to test directories
    for well in healthy_wells[:num_test_healthy]:
        for i in range(4):
            for j in range(4):
                try:

                    img = prefix + well + f"_{i}_{j}.TIF"
                    src = os.path.join(healthy_dir, img)
                    dst = os.path.join(test_healthy_dir, img)
                    shutil.move(src, dst)
                    splitted_images.append((img, "test"))
                except Exception as e:
                    print("Error with the following file during splitting to test:", img)
                    print(repr(e))
    for well in unhealthy_wells[:num_test_unhealthy]:
        for i in range(4):
            for j in range(4):
                try:

                    img = prefix + well + f"_{i}_{j}.TIF"
                    src = os.path.join(unhealthy_dir, img)
                    dst = os.path.join(test_unhealthy_dir, img)
                    shutil.move(src, dst)
                    splitted_images.append((img, "test"))
                except Exception as e:
                    print("Error with the following file during splitting to test:", img)
                    print(repr(e))

    # Move remaining images to training directories
    for well in healthy_wells[num_test_healthy:]:
        for i in range(4):
            for j in range(4):
                try:

                    img = prefix + well + f"_{i}_{j}.TIF"
                    src = os.path.join(healthy_dir, img)
                    dst = os.path.join(training_healthy_dir, img)
                    shutil.move(src, dst)
                    splitted_images.append((img, "train"))
                except Exception as e:
                    print("Error with the following file during splitting to test:", img)
                    print(repr(e))
    for well in unhealthy_wells[num_test_unhealthy:]:
        for i in range(4):
            for j in range(4):

                try:

                    img = prefix + well + f"_{i}_{j}.TIF"
                    src = os.path.join(unhealthy_dir, img)
                    dst = os.path.join(training_unhealthy_dir, img)
                    shutil.move(src, dst)
                    splitted_images.append((img, "train"))

                except Exception as e:
                    print("Error with the following file during splitting to test:", img)
                    print(repr(e))


    # Write the csv file

    with open(index_file, mode="w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image", "Category"])
        csv_writer.writerows(splitted_images)



    print("Images are splitted.")

