import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# defining global variable path
image_path = "C:/Users/eddie/Documents/233-drucker-pc1_200709170002"

output = "C:/Users/eddie/Documents/NPC_3/train_set/"

'''function to load folder into arrays and 
then it returns that same array'''
def loadImages(path):
	# Put files into lists and return them as one list of size 4
	image_files = sorted([os.path.join(path, file)
		for file in os.listdir(path) if "d0.TIF" in file])
 
	return image_files


# Display one image
def display_one(a, title1 = "Original"):
	plt.imshow(a), plt.title(title1)
	plt.xticks([]), plt.yticks([])
	plt.show()

# Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
	plt.subplot(121), plt.imshow(a), plt.title(title1)
	plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(b), plt.title(title2)
	plt.xticks([]), plt.yticks([])
	plt.show()

# Preprocessing
def processing(data):
	# loading image
	
	print('Original size',cv2.imread(data[0]).shape)

	#res_img = []

	for i, img in enumerate(data):
		try:

			base = os.path.basename(img)

			well = base[-11:-9]

			wt = True if int(well) <=6 else False

			ch0 = cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2GRAY)
			ch1 = cv2.cvtColor(cv2.imread(img.replace("d0", "d1")),cv2.COLOR_BGR2GRAY)
			ch2 = cv2.cvtColor(cv2.imread(img.replace("d0", "d2")),cv2.COLOR_BGR2GRAY)
			merge = np.zeros((ch0.shape[0], ch0.shape[1], 3))

			merge [:,:,0] = ch0
			merge [:,:,1] = ch1
			merge [:,:,2] = ch2

			# --------------------------------
			# setting dim of the resize
			height = 128
			width = 128

			margin = 34

			dim = (width, height)

			res = cv2.resize(merge, dim, interpolation=cv2.INTER_LINEAR)

			#cv2.imwrite(output + str(i) + "A.png", res)

			res = res[margin:-margin, margin:-margin]

			#cv2.imwrite(output + str(i) + "B.png", res)

			# normalize

			res = cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)

			#cv2.imwrite(output + str(i) + "C.png", res)

			#res_img.append(res)

			if wt:
				cv2.imwrite(output + "healthy/"+ os.path.splitext(base)[0] + ".png", res)
			else:
				cv2.imwrite(output + "unhealthy/"+ os.path.splitext(base)[0] + ".png", res)

		except Exception as e:
			print()
			print("Exception occured with following : " + img)
			print(repr(e))


def main():
	# calling global variable
	global image_path
	'''The var Dataset is a list with all images in the folder '''          
	dataset = loadImages(image_path)

	print("List of files the first 3 in the folder:\n",dataset[:3])
	print("--------------------------------")

	# sending all the images to pre-processing
	pro = processing(dataset)



main()