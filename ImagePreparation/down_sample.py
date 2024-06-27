import os
import cv2


def start_downsample(images_dir):

    images = os.listdir(images_dir)

    
   
    height = 64
    width = 64
    dim = (width, height)
    for img_file in images:
        if img_file[-3:] == "TIF":            
            try:
                img = cv2.cvtColor(cv2.imread(os.path.join(images_dir, img_file)), cv2.COLOR_BGR2GRAY)

                res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

                # normalize

                res = cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)

                cv2.imwrite(os.path.join(images_dir, img_file), res)

            except Exception as e:
                print()
                print("Exception occured with following during down sampling: " + img_file)
                print(repr(e))


    print("Done down sampling.")
