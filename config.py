import os

DIR_NAME = os.path.dirname(__file__)


model_id = 1
c = 1
n_rounds = 100000
step_arms = 0.2
step_overhead = 0.1
cuda = True
dim = 300
resize_img = 330
model_name = "mobilenet"
n_branches = 1
dataset_name = "caltech256"
distortion_type = "pristine"
exit_type = "bnpool"
distribution = "predefined"
pretrained = True
seed = 42
batch_size_train = 64
split_ratio = 0.1

nr_class_dict = {"caltech256": 257}

dataset_path_dict = {"caltech256": os.path.join(DIR_NAME, "datasets", "caltech256", "257_ObjectCategories"),
"cifar10": os.path.join(DIR_NAME,"datasets", "cifar10") }

indices_path_dict = {"caltech256": os.path.join(DIR_NAME, "indices", "caltech256"),
"cifar10": os.path.join(DIR_NAME,"indices", "cifar10") }

distortion_lvl_dict = {"pristine": [0], "gaussian_blur": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5], 
"gaussian_noise": [5, 10, 20, 30, 40]}

fontsize = 16