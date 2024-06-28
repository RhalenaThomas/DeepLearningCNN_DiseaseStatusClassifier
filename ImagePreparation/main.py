from split_and_organize import organize, split
from QC import start_QC
from cropImages import crop
from merge import start_merge
from down_sample import start_downsample
healthy_col = ["01", "03", "05"]
unhealthy_col = ["07", "09", "11"]

images_folder = "./images/example/"

parent_dir = "./images/example_processed/"


#crop(images_dir=images_folder)


#start_QC(images_dir=images_folder)

#start_downsample(images_dir=images_folder)

#start_merge(images_dir=images_folder)

healthy_wells, unhealthy_wells = organize(images_folder, parent_dir, healthy_col, unhealthy_col)

prefix = "233-drucker-pc1_201017100002_"
split(parent_dir, healthy_wells, unhealthy_wells, "index_file.csv", prefix)
