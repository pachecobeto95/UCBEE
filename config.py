import os

DIR_NAME = os.path.dirname(__file__)


model_id = 1
c = 1
n_rounds = 100000
step_arms = 0.2
step_overhead = 0.1
cuda = True
dim = 224
model_name = "alexnet"
n_branches = 1
dataset_name = "caltech256"

nr_class_dict = {"caltech256": 257}

dataset_path_dict = {"caltech256": os.path.join(DIR_NAME, "datasets", "caltech256", "256_ObjectCategories"),
"cifar10": os.path.join(DIR_NAME,"datasets", "cifar10") }

indices_path_dict = {"caltech256": os.path.join(DIR_NAME, "indices", "caltech256"),
"cifar10": os.path.join(DIR_NAME,"indices", "cifar10") }

distortion_type = "pristine"