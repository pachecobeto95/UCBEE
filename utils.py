from torchvision import datasets, transforms
import torch, os, sys, requests, random, logging, torchvision, config, cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import b_mobilenet

class DistortionApplier(object):
	def __init__(self, distortion_function, distortion_lvl):

		self.distortion_lvl = distortion_lvl
		self.distortion_function = getattr(self, distortion_function, self.distortion_not_found)

	def __call__(self, img):
		return self.distortion_function(img, self.distortion_lvl)

	def gaussian_blur(self, img, distortion_lvl):
		image = np.array(img)
		#blurred_img = cv2.GaussianBlur(image, (4*distortion_lvl+1, 4*distortion_lvl+1), distortion_lvl, None, sigma, cv2.BORDER_CONSTANT)
		#return Image.fromarray(blurred_img) 
		kernel_size = (4*distortion_lvl+1, 4*distortion_lvl+1)
		blurrer = transforms.GaussianBlur(kernel_size=kernel_size, sigma=distortion_lvl)
		return blurrer(img)

	def gaussian_noise(self, img, distortion_lvl):
		image = np.array(img)
		noise_img = image + np.random.normal(0, distortion_lvl, (image.shape[0], image.shape[1], image.shape[2]))
		return Image.fromarray(np.uint8(noise_img)) 

	def pristine(self, img, distortion_lvl):
		return img

	def motion_blur(self, img, distortion_lvl):

		img = np.array(img)

		# generating the kernel
		kernel_motion_blur = np.zeros((distortion_lvl, distortion_lvl))
		kernel_motion_blur[int((distortion_lvl-1)/2), :] = np.ones(distortion_lvl)
		kernel_motion_blur = kernel_motion_blur / distortion_lvl

		# applying the kernel to the input image
		blurred_img = cv2.filter2D(img, -1, kernel_motion_blur)

		return blurred_img

	def distortion_not_found(self):
		raise Exception("This distortion type has not implemented yet.")


def get_indices_caltech256(dataset, split_ratio):
	
	nr_samples = len(dataset)
	#print("Nr Samples: %s"%(nr_samples))
	indices = list(range(nr_samples))
	np.random.shuffle(indices)

	train_val_size = nr_samples - int(np.floor(split_ratio * nr_samples))


	train_val_idx, test_idx = indices[:train_val_size], indices[train_val_size:]
	#print("Train Indices: %s"%(train_val_idx))

	#np.random.shuffle(train_val_idx)

	#train_size = len(train_val_idx) - int(np.floor(split_ratio * len(train_val_idx) ))
	#print("Train Size: %s"%(train_size))
	#train_idx, val_idx = train_val_idx[:train_size], train_val_idx[train_size:]

	#print("Train Indices: %s"%(train_idx))
	#sys.exit()
	return train_val_idx, test_idx, test_idx


def load_caltech256(args, dataset_path, save_indices_path, distortion_lvl):
	mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	torch.manual_seed(args.seed)
	np.random.seed(seed=args.seed)


	transformations_train = transforms.Compose([
		#transforms.Resize((args.input_dim, args.input_dim)),
		transforms.Resize((config.resize_img, config.resize_img)),
		transforms.CenterCrop((config.dim, config.dim)),		
		transforms.RandomChoice([
			transforms.ColorJitter(brightness=(0.80, 1.20)),
			transforms.RandomGrayscale(p = 0.25)]),
		transforms.RandomHorizontalFlip(p=0.25),
		transforms.RandomRotation(25),
		transforms.RandomApply([DistortionApplier(args.distortion_type, distortion_lvl)], p=1),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize((config.resize_img, config.resize_img)),
		transforms.CenterCrop((config.dim, config.dim)),		
		transforms.RandomApply([DistortionApplier(args.distortion_type, distortion_lvl)], p=1),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	# This block receives the dataset path and applies the transformation data. 
	train_set = datasets.ImageFolder(dataset_path, transform=transformations_train)


	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	train_idx_path = os.path.join(save_indices_path, "training_idx_caltech256_3_branches_%s.npy"%(args.model_id))
	val_idx_path = os.path.join(save_indices_path, "validation_idx_caltech256_3_branches_%s.npy"%(args.model_id))
	#test_idx_path = os.path.join(save_indices_path, "test_idx_caltech256.npy")

	if( os.path.exists(train_idx_path) ):
		#Load the indices to always use the same indices for training, validating and testing.
		train_idx = np.load(train_idx_path)
		val_idx = np.load(val_idx_path)
		#test_idx = np.load(test_idx_path)

	else:
		# This line get the indices of the samples which belong to the training dataset and test dataset. 
		train_idx, val_idx, test_idx = get_indices_caltech256(train_set, config.split_ratio)

		#Save the training, validation and testing indices.
		np.save(train_idx_path, train_idx), np.save(val_idx_path, val_idx)#, np.save(test_idx_path, test_idx)

	train_data = torch.utils.data.Subset(train_set, indices=train_idx)
	val_data = torch.utils.data.Subset(val_set, indices=val_idx)
	#test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=4, pin_memory=True)
	#test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, pin_memory=True)

	return train_loader, val_loader, val_loader


def get_indices(dataset, split_ratio):
	
	nr_samples = len(dataset)
	indices = list(range(nr_samples))
	np.random.shuffle(indices)

	train_size = nr_samples - int(np.floor(split_ratio * nr_samples))


	train_idx, test_idx = indices[:train_val_size], indices[train_val_size:]

	return train_idx, test_idx 

def load_cifar10(args, dataset_path, indices_path, distortion_values):


	mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	torch.manual_seed(args.seed)
	np.random.seed(seed=args.seed)

	transformations_train = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.CenterCrop((224, 224)),
		transforms.RandomApply([DistortionApplier(args.distortion_type, distortion_lvl)], p=1),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.CenterCrop((224, 224)),
		transforms.RandomApply([DistortionApplier(args.distortion_type, distortion_lvl)], p=1),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])


	train_val_data = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transformations_train)

	train_idx_path = os.path.join(save_indices_path, "training_idx_cifar10.npy")
	val_idx_path = os.path.join(save_indices_path, "validation_idx_cifar10.npy")
	test_idx_path = os.path.join(save_indices_path, "test_idx_cifar10.npy")

	if( os.path.exists(train_idx_path) ):
		#Load the indices to always use the same indices for training, validating and testing.
		train_idx = np.load(train_idx_path)
		val_idx = np.load(val_idx_path)
		test_idx = np.load(test_idx_path)

	else:
		# This line get the indices of the samples which belong to the training dataset and test dataset. 
		train_idx, val_idx = get_indices(train_val_data, args.split_ratio)

		#Save the training, validation and testing indices.
		np.save(train_idx_path, train_idx), np.save(val_idx_path, val_idx)

	train_data = torch.utils.data.Subset(train_val_data, indices=train_idx)
	val_data = torch.utils.data.Subset(train_val_data, indices=val_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

	#Downloading test data
	test_data = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)

	test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

	return train_loader, val_loader, test_loader


def load_model(model, modelPath, device):
	model.load_state_dict(torch.load(modelPath, map_location=device)["model_state_dict"])	
	return model.to(device)

def init_b_mobilenet(modelPath):
	n_classes = 258
	img_dim = 300
	exit_type = None
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	pretrained = False
	n_branches = 3

	b_mobilenet_pristine = b_mobilenet.B_MobileNet(n_classes, pretrained, n_branches, img_dim, exit_type, device)

	pristine_model = load_model(b_mobilenet_pristine, modelPath, device)

	return b_mobilenet_pristine