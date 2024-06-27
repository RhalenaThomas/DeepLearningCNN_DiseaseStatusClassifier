import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import os
import shutil
from skimage.io import imread
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.utilities.java
import pathlib
import pandas as pd



def remove_blurry_images(images_dir):
    sigma = 2
    thresh = 150
    thresh_count = 1000
    flagged = {}
    excluded_img = {}

    os.makedirs(os.path.join(images_dir, "flagged_images"), exist_ok=True)
    images = os.listdir(images_dir)

    for image in images :
        img_filename = os.path.join(images_dir, image)

            
        if img_filename[-3:] == "TIF":
            img = imread(img_filename)

            img_smooth = ndi.gaussian_filter(img, sigma)

            thresh_img  = img_smooth > thresh

            unique, counts = np.unique(thresh_img, return_counts = True)

            result = dict(zip(unique, counts))
            #print(result, image)
            if True in result.keys() and result[True] >= thresh_count:
                #shutil.move(img_filename, os.path.join(images_dir, "flagged_images"))

                img_info = image.split("_")
                _, well, _, _ = img_info[1:5]
                channel = well[-2:]
                image_name = "_".join(i for i in img_info[1:5])
                image_name = image_name.replace(channel, "")

                if image_name in flagged.keys():
                    flagged[image_name] += 1
                    image = image.replace(channel, "d0")
                    excluded_img[image] = True

                else:
                    flagged[image_name] = 1

    for image in excluded_img.keys():
        img_filename = os.path.join(images_dir, image)

        shutil.move(img_filename, os.path.join(images_dir, "flagged_images"))
        shutil.move(img_filename.replace("d0", "d1"), os.path.join(images_dir, "flagged_images"))
        shutil.move(img_filename.replace("d0", "d2"), os.path.join(images_dir, "flagged_images"))
        


    print("Excluded images are:", excluded_img.keys())



def run_cell_profilerPipeline(pipeline_file):
    cellprofiler_core.preferences.set_headless()
    cellprofiler_core.utilities.java.start_java()
    pipeline = cellprofiler_core.pipeline.Pipeline()

    pipeline.load(pipeline_file)
    print("Pipeline loaded.")

    cellprofiler_core.preferences.set_default_output_directory(images_dir)

    file_list = list(pathlib.Path('.').absolute().glob(f'{images_dir}/*.TIF'))
    files = [file.as_uri() for file in file_list]
    print("Number of files:", len(files))

    pipeline.read_file_list(files)

    output_measurements = pipeline.run()
    cellprofiler_core.utilities.java.stop_java()


def qc_nuclei_count(csv_file, images_dir):
    lower_bound = 2
    upper_bound = 40


    df_img = pd.read_csv(csv_file)

    filtered_img = df_img[df_img['Count_Nucleus'].between(lower_bound, upper_bound)]
    img_not_passed_filter = df_img[~df_img['Count_Nucleus'].between(lower_bound, upper_bound)]

    filtered_img.to_csv("filtered_img.csv")
    img_not_passed_filter.to_csv("excluded_images.csv")

    for row in img_not_passed_filter.iterrows():
        filename_hoechst = row["FileName_Hoechst"]
        filename_WGA = row["FileName_WGA"]
        filename_mito = row["FileName_mitotracker"]

        shutil.move(os.path.join(images_dir, filename_hoechst), os.path.join(images_dir, "flagged_images"))
        shutil.move(os.path.join(images_dir, filename_WGA), os.path.join(images_dir, "flagged_images"))
        shutil.move(os.path.join(filename_mito, filename_hoechst), os.path.join(images_dir, "flagged_images"))

def start_QC(images_dir):
    cellProfiler_pipeline = './QC3.cppipe'
    csv_nuclei = images_dir + "/" + "MyExpt_Image.csv"


    remove_blurry_images(images_dir)

    print("Finished the first QC.")

    run_cell_profilerPipeline(cellProfiler_pipeline)

    print("Finished running the pipeline.")

    qc_nuclei_count(csv_nuclei, images_dir)

    print("Finished filtering based on the cell profiler pipeline.")






