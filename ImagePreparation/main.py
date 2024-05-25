from split_and_organize import organize, split

healthy_col = ["01", "03", "05"]
unhealthy_col = ["07", "09", "11"]

images_folder = "./images/data/233-drucker-pc1_201017100002/"

parent_dir = "./images/data_24may2024/"


healthy_triples, unhealthy_triples = organize(images_folder, parent_dir, healthy_col, unhealthy_col)

prefix = "233-drucker-pc1_201017100002_"
split(parent_dir, healthy_triples, unhealthy_triples, "index_file.csv", prefix)
