import os
import cv2
import numpy as np

images_dir = "./images/exampleImages"


def start_merge(images_dir):
    images = os.listdir(images_dir)

    for img in images:

        try: 
            if img[-3:] == "TIF":
                img_info = img.split("_")
                well = img_info[2]
                channel = well[-2:]

                if channel == "d0":
                    
                    img_filename = os.path.join(images_dir, img)
                    ch0 = cv2.cvtColor(cv2.imread(img_filename),cv2.COLOR_BGR2GRAY)
                    ch1 = cv2.cvtColor(cv2.imread(img_filename.replace("d0", "d1")),cv2.COLOR_BGR2GRAY)
                    ch2 = cv2.cvtColor(cv2.imread(img_filename.replace("d0", "d2")),cv2.COLOR_BGR2GRAY)

                    merge = np.zeros((ch0.shape[0], ch0.shape[1], 3))

                    merge[:,:,0] = ch1
                    merge[:,:,1] = ch2
                    merge[:,:,2] = ch2

                    cv2.imwrite(img_filename.replace("d0", ""), merge)

                    for i in range(3):
                        os.remove(img_filename.replace("d0", f"d{i}"))

        except Exception as e:
            print("Error with the following file during merging:", img)
            print(repr(e))


    print("Finished merging")
