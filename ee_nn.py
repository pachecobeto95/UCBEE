import torchvision, torch, os, sys, time, math
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pthflops import count_ops
from torch import Tensor

"""
(18): ConvBNActivation(
      (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=1000, bias=True)
"""
"""
class EarlyExitBlock(nn.Module):
	def __init__(self, input_shape, last_channel, n_classes, exit_type, device):
		super(EarlyExitBlock, self).__init__()
		self.input_shape = input_shape
		_, channel, width, height = input_shape
		
		self.layers = nn.ModuleList()

		if (exit_type == 'bnpool'):
			self.layers.append(nn.BatchNorm2d(channel))
			self.layers.append(nn.AdaptiveAvgPool2d(1))

		elif(exit_type == 'conv'):
			self.layers.append(nn.Conv2d(channel, last_channel, kernel_size=(1,1)))
			self.layers.append(nn.BatchNorm2d(last_channel))
			self.layers.append(nn.MaxPool2d(2))
			self.layers.append(nn.ReLU6(inplace=True))

		elif(exit_type == 'pooling'):
			self.layers.append(nn.BatchNorm2d(channel))
			self.layers.append(nn.MaxPool2d(2))

		else:
			self.layers = nn.ModuleList()

		#This line defines the data shape that fully-connected layer receives.
		current_channel, current_width, current_height = self.get_current_data_shape()
		self.classifier = nn.Sequential(nn.Dropout(0.2), 
			nn.Linear(current_channel*current_width*current_height, n_classes))
		self.classifier = self.classifier.to(device)

	def get_current_data_shape(self):
		print("Before")
		print(self.input_shape)
		_, channel, width, height = self.input_shape
		temp_layers = nn.Sequential(*self.layers)

		input_tensor = torch.rand(1, channel, width, height)
		_, output_channel, output_width, output_height = temp_layers(input_tensor).shape
		print("After")
		print(output_channel, output_width, output_height)		
		return output_channel, output_width, output_height

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		#x = x.view(x.size(0), -1)
		x = torch.flatten(x, 1)
		output = self.classifier(x)

		return output
"""

class EarlyExitBlock(nn.Module):
	def __init__(self, input_shape, pool_size, n_classes, exit_type, device):
		super(EarlyExitBlock, self).__init__()
		self.input_shape = input_shape

		_, channel, width, height = input_shape
		self.expansion = width * height if exit_type == 'plain' else 1

		self.layers = nn.ModuleList()

		if (exit_type == 'bnpool'):
			self.layers.append(nn.BatchNorm2d(channel))

		if (exit_type != 'plain'):
			self.layers.append(nn.AdaptiveAvgPool2d(1))

		#This line defines the data shape that fully-connected layer receives.
		current_channel, current_width, current_height = self.get_current_data_shape()

		self.layers = self.layers#.to(device)

		#This line builds the fully-connected layer
		self.classifier = nn.Sequential(nn.Linear(current_channel*current_width*current_height, n_classes))#.to(device)

		self.softmax_layer = nn.Softmax(dim=1)


	def get_current_data_shape(self):
		_, channel, width, height = self.input_shape
		temp_layers = nn.Sequential(*self.layers)

		input_tensor = torch.rand(1, channel, width, height)
		_, output_channel, output_width, output_height = temp_layers(input_tensor).shape
		return output_channel, output_width, output_height

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		x = x.view(x.size(0), -1)
		output = self.classifier(x)
		#confidence = self.softmax_layer()
		return output


class Early_Exit_DNN(nn.Module):
	def __init__(self, model_name: str, n_classes: int, pretrained: bool, n_branches: int, input_dim: int, 
		device, exit_type: str, distribution: str, ee_point_location=10):

		super(Early_Exit_DNN, self).__init__()

		"""
		This classes builds an early-exit DNNs architectures
		Args:

		model_name: model name 
		n_classes: number of classes in a classification problem, according to the dataset
		pretrained: 
		n_branches: number of branches (early exits) inserted into middle layers
		input_dim: dimension of the input image
		exit_type: type of the exits
		distribution: distribution method of the early exit blocks.
		device: indicates if the model will processed in the cpu or in gpu
		    
		Note: the term "backbone model" refers to a regular DNN model, considering no early exits.
		"""

		self.model_name = model_name
		self.n_classes = n_classes
		self.pretrained = pretrained
		self.n_branches = n_branches
		self.input_dim = input_dim
		self.exit_type = exit_type
		self.distribution = distribution
		self.device = device
		self.ee_point_location = ee_point_location

		build_early_exit_dnn = self.select_dnn_architecture_model()
		build_early_exit_dnn()

	def select_dnn_architecture_model(self):
		"""
		This method selects the backbone to insert the early exits.
		"""

		architecture_dnn_model_dict = {"mobilenet": self.early_exit_mobilenet, 
		"alexnet": self.early_exit_alexnet}
		print(self.model_name)
		return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)


	def select_distribution_method(self):
		"""
		This method selects the distribution method to insert early exits into the middle layers.
		"""
		distribution_method_dict = {"linear":self.linear_distribution,
		"pareto":self.paretto_distribution,
		"fibonacci":self.fibo_distribution}
		return distribution_method_dict.get(self.distribution, self.invalid_distribution)
    
	def linear_distribution(self, i):
		"""
		This method defines the Flops to insert an early exits, according to a linear distribution.
		"""
		flop_margin = 1.0 / (self.n_branches+1)
		return self.total_flops * flop_margin * (i+1)

	def paretto_distribution(self, i):
		"""
		This method defines the Flops to insert an early exits, according to a pareto distribution.
		"""
		return self.total_flops * (1 - (0.8**(i+1)))

	def fibo_distribution(self, i):
		"""
		This method defines the Flops to insert an early exits, according to a fibonacci distribution.
		"""
		gold_rate = 1.61803398875
		return total_flops * (gold_rate**(i - self.num_ee))

	def verifies_nr_exits(self, backbone_model):
		"""
		This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
		"""
    
		total_layers = len(list(backbone_model.children()))
		if (self.n_branches >= total_layers):
			raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")


	def where_insert_early_exits(self):
		"""
		This method defines where insert the early exits, according to the dsitribution method selected.
		Args:

		total_flops: Flops of the backbone (full) DNN model.
		"""
		threshold_flop_list = []
		distribution_method = self.select_distribution_method()

		for i in range(self.n_branches):
			threshold_flop_list.append(distribution_method(i))

		return threshold_flop_list

	def invalid_model(self):
		raise Exception("This DNN backbone model has not implemented yet.")

	def invalid_distribution(self):
		raise Exception("This early-exit distribution has not implemented yet.")

	def countFlops(self, model):
		input_data = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
		flops, all_data = count_ops(model, input_data, print_readable=False, verbose=False)
		return flops

	def is_suitable_for_exit(self, nr_block):

		if(self.distribution=="predefined"):
			return (self.stage_id < self.n_branches) and (nr_block > self.ee_point_location)

		else:
			intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers))).to(self.device)
			x = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
			current_flop, _ = count_ops(intermediate_model, x, verbose=False, print_readable=False)
			return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

	def add_exit_block(self):
		"""
		This method adds an early exit in the suitable position.
		"""
		input_tensor = torch.rand(1, 3, self.input_dim, self.input_dim)

		self.stages.append(nn.Sequential(*self.layers))
		x = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)

		feature_shape = nn.Sequential(*self.stages)(x).shape
		
		self.exits.append(EarlyExitBlock(feature_shape, self.last_channel, self.n_classes, self.exit_type, self.device).to(self.device))
		self.layers = nn.ModuleList()
		self.stage_id += 1    

	def early_exit_alexnet(self):

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		self.stage_id = 0

		self.last_channel = 1280


		# Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
		backbone_model = models.alexnet(self.pretrained).to(self.device)

		#backbone_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=self.pretrained)

		# It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
		self.verifies_nr_exits(backbone_model.features)

		# This obtains the flops total of the backbone model
		self.total_flops = self.countFlops(backbone_model)

		# This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		#self.threshold_flop_list = self.where_insert_early_exits()

		# This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		if(self.distribution != "predefined"):
			self.threshold_flop_list = self.where_insert_early_exits()


		for nr_block, block in enumerate(backbone_model.features.children()):
			
			self.layers.append(block)
			if (self.is_suitable_for_exit(nr_block)):
				self.add_exit_block()

		self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6,6)))
		self.stages.append(nn.Sequential(*self.layers))

		self.classifier = backbone_model.classifier
		
		self.classifier[4] = nn.Linear(in_features=4096, out_features=1024, bias=True)
		self.classifier[6] = nn.Linear(in_features=1024, out_features=self.n_classes, bias=True)
		
		self.softmax = nn.Softmax(dim=1)




	def early_exit_mobilenet(self):

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		self.stage_id = 0

		self.last_channel = 1280


		# Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
		backbone_model = models.mobilenet_v2(self.pretrained).to(self.device)

		# It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
		self.verifies_nr_exits(backbone_model.features)

		# This obtains the flops total of the backbone model
		self.total_flops = self.countFlops(backbone_model)

		# This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		#self.threshold_flop_list = self.where_insert_early_exits()

		# This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		if(self.distribution != "predefined"):
			self.threshold_flop_list = self.where_insert_early_exits()

		for nr_block, block in enumerate(backbone_model.features.children()):
			
			self.layers.append(block)
			if (self.is_suitable_for_exit(nr_block)):
				self.add_exit_block()

		self.layers.append(nn.AdaptiveAvgPool2d(1))
		self.stages.append(nn.Sequential(*self.layers))

		#self.classifier = backbone_model.classifier
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(self.last_channel, self.n_classes),)
		
		self.softmax = nn.Softmax(dim=1)


	def forwardTrain(self, x):
		"""
		This method is used to train the early-exit DNN model
		"""

		output_list, conf_list, class_list  = [], [], []

		for i, exitBlock in enumerate(self.exits):

			x = self.stages[i](x)
			output_branch = exitBlock(x)

			#Confidence is the maximum probability of belongs one of the predefined classes and inference_class is the argmax
			conf, infered_class = torch.max(self.softmax(output_branch), 1)
			
			output_list.append(output_branch), conf_list.append(conf), class_list.append(infered_class)

		x = self.stages[-1](x)

		x = torch.flatten(x, 1)

		output = self.classifier(x)
		infered_conf, infered_class = torch.max(self.softmax(output), 1)
		output_list.append(output), conf_list.append(infered_conf), class_list.append(infered_class)

		return output_list, conf_list, class_list

	def forwardInferenceNoCalib(self, x):
		output_list, conf_list, infered_class_list = [], [], []

		for i, exitBlock in enumerate(self.exits):
			x = self.stages[i](x)

			output_branch = exitBlock(x)
			conf, infered_class = torch.max(self.softmax(output_branch), 1)

			output_list.append(output_branch), infered_class_list.append(infered_class), conf_list.append(conf)

		x = self.stages[-1](x)
		x = torch.flatten(x, 1)

		output = self.classifier(x)

		conf, infered_class = torch.max(self.softmax(output), 1)

		output_list.append(output), infered_class_list.append(infered_class), conf_list.append(conf)

		return output_list, conf_list, infered_class_list

	def global_temperature_scaling(self, logits, temp_overall):
		return torch.div(logits, temp_overall)

	def per_branch_temperature_scaling(self, logits, temp, exit_branch):
		return torch.div(logits, temp[exit_branch])

	def forwardGlobalCalibration(self, x, temp_overall):
		output_list, conf_list, class_list = [], [], []

		n_exits = self.n_branches + 1

		for i, exitBlock in enumerate(self.exits):
			x = self.stages[i](x)
			output_branch = exitBlock(x)

			output_branch = self.global_temperature_scaling(output_branch, temp_overall)

			conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)

			output_list.append(output_branch), conf_list.append(conf_branch), class_list.append(infered_class_branch)

		x = self.stages[-1](x)

		x = torch.flatten(x, 1)

		output = self.classifier(x)
		output = self.global_temperature_scaling(output, temp_overall)

		conf, infered_class = torch.max(self.softmax(output), 1)
		output_list.append(output), conf_list.append(conf), class_list.append(infered_class)

		return output_list, conf_list, class_list

	def forwardPerBranchesCalibration(self, x, temp_branches):
		output_list, conf_list, class_list = [], [], []
		n_exits = self.n_branches + 1

		for i, exitBlock in enumerate(self.exits):
			x = self.stages[i](x)
			output_branch = exitBlock(x)
			output_branch = self.per_branch_temperature_scaling(output_branch, temp_branches, i)

			conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)

			output_list.append(output_branch), conf_list.append(conf_branch), class_list.append(infered_class_branch)

		x = self.stages[-1](x)

		x = torch.flatten(x, 1)

		output = self.classifier(x)
		output = self.per_branch_temperature_scaling(output, temp_branches, -1)

		conf, infered_class = torch.max(self.softmax(output), 1)
		output_list.append(output), conf_list.append(conf), class_list.append(infered_class)

		return output_list, conf_list, class_list

	def forwardAllSamplesCalibration(self, x, temp):
		output_list, conf_list, class_list = [], [], []
		n_exits = self.n_branches + 1

		for i, exitBlock in enumerate(self.exits):
			x = self.stages[i](x)
			output_branch = exitBlock(x)
			output_branch = self.per_branch_temperature_scaling(output_branch, temp, i)

			conf_branch, infered_class_branch = torch.max(self.softmax(output_branch), 1)

			output_list.append(output_branch), conf_list.append(conf_branch), class_list.append(infered_class_branch)

		x = self.stages[-1](x)

		x = torch.flatten(x, 1)

		output = self.classifier(x)
		output = self.per_branch_temperature_scaling(output, temp, -1)
		conf, infered_class = torch.max(self.softmax(output), 1)

		output_list.append(output), conf_list.append(conf), class_list.append(infered_class)

		return output_list, conf_list, class_list