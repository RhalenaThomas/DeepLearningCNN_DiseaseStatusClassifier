import os


def organize(images_folder, parent_dir, healthy_columns, unhealthy_columns):

    '''
    Given a set of images, class them into healthy and unhealthy folder depending on their column
    return a list of healthy triples (row, column, field) and triple of unhealthy
    '''

    healthy_triples = []
    unhealthy_triples = []

    healthy_dir = os.path.join(parent_dir, 'healthy')
    unhealthy_dir = os.path.join(parent_dir, 'unhealthy')

    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)

    images_files = os.listdir(images_folder)



    for file in image_files:

        if "TIF" in file:
            filename_split = file.split("_")
            col = filename_split[-1][1:3]
            row = filename_split[-1][0]
            field = "f"
            for c in filename_split[-1][4:7]:
                print(c)
                if c.isdigit():
                    field = field + c
                else:
                    break

            if col in healthy_columns:
                folder = os.path.join(healthy_dir, filename)
                shutil.copy(image_path, folder)
                healthy_triples.append((row, col, field))
            elif col in unhealthy_columns :
                folder = os.path.join(unhealthy_dir, filename)
                shutil.copy(image_path, folder)
                unhealthy_triples.append((row, col, field))



    print("Images are organized into healthy and unhealthy directories.")
    return healthy_triples, unhealthy_triples



def split(parent_dir, test_ratio=0.3, healthy_triples, unhealthy_triples, index_file):

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


    # Shuffle the lists of healthy and unhealthy triples
    random.shuffle(healthy_triples)
    random.shuffle(unhealthy_triples)



    # Calculate the number of images for the test set based on the test_ratio
    # Divide by 3 since all the three channels should be in the same folder
    num_test_healthy = int(len(healthy_triples) * test_ratio) / 3
    num_test_unhealthy = int(len(unhealthy_triples) * test_ratio) / 3

    splitted_images = []

    # Move images from healthy and unhealthy directories to test directories
    for row, col, field in healthy_images[:num_test_healthy]:
        for i in range(3):
            img = os.path.join(prefix, row+col+field+f"d{i}.TIF")
            src = os.path.join(healthy_dir, img)
            dst = os.path.join(test_healthy_dir, img)
            shutil.move(src, dst)
            splitted_images.append(img, "test")
    for img in unhealthy_images[:num_test_unhealthy]:
        for i in range(3):
            img = os.path.join(prefix, row+col+field+f"d{i}.TIF")
            src = os.path.join(healthy_dir, img)
            dst = os.path.join(test_healthy_dir, img)
            shutil.move(src, dst)
            splitted_images.append(img, "test")

    # Move remaining images to training directories
    for img in healthy_images[num_test_healthy:]:
        for i in range(3):
            img = os.path.join(prefix, row+col+field+f"d{i}.TIF")
            src = os.path.join(healthy_dir, img)
            dst = os.path.join(test_healthy_dir, img)
            shutil.move(src, dst)
            splitted_images.append(img, "train")
    for img in unhealthy_images[num_test_unhealthy:]:
        for i in range(3):
            img = os.path.join(prefix, row+col+field+f"d{i}.TIF")
            src = os.path.join(healthy_dir, img)
            dst = os.path.join(test_healthy_dir, img)
            shutil.move(src, dst)
            splitted_images.append(img, "train")




    # Write the csv file

    with open(index_file, mode="w", newline='') as csv_file:
        csv_writer = csv_writer(csv_file)
        csv_writer.writerow(["Image", "Category"])
        csv_writer.writerows(splitted_images)



    print("Images are splitted.")
