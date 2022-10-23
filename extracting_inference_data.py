import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import itertools, argparse, os, sys, random, logging, config, torch, torchvision, ee_nn, utils


def run_inference_data(model, test_loader, p_tar, n_branches, calib_type, distortion_type, distortion_lvl, device):

	df_result = pd.DataFrame()

	n_exits = n_branches + 1
	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, exit_branch_list, id_list = [], [], []

	model.eval()

	with torch.no_grad():
		for i, (data, target) in tqdm(test_loader):

			data, target = data.to(device), target.to(device)

			if (calib_type == "no_calib"):
				_, conf_branches, infered_class_branches = model.forwardInferenceNoCalib(data)

			elif(calib_type == "global_calib"):
				_, conf_branches, infered_class_branches = model.forwardGlobalCalibration(data)

			elif(calib_type == "per_branch_calib"):
				_, conf_branches, infered_class_branches = model.forwardPerBranchesCalibration(data)
			
			else:
				 _, conf_branches, infered_class_branches = model.forwardAllSamplesCalibration(data)

			conf_branches_list.append([conf.item() for conf in conf_branches])
			infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
			#id_list.append(i)
			target_list.append(target.item())

			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)

	#results = {"distortion_type": distortion_type, "distortion_lvl": distortion_lvl, "p_tar": [p_tar]*len(target_list), 
	#"target": target_list, "id": id_list}
	results = {"distortion_type": [distortion_type]*len(target_list), "distortion_lvl": [distortion_lvl]*len(target_list), 
	"p_tar": [p_tar]*len(target_list), "target": target_list}
	
	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i]})

	return results


def save_result(result, save_path):
	df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df_result = df_result.append(df)
	df_result.to_csv(save_path)


def extracting_inference_data(model, p_tar_list, distortion_lvl_list, inference_data_path, dataset_path, indices_path, calib_type, distortion_type, device):

	for distortion_lvl in distortion_lvl_list:
		print("Distortion Level: %s"%(distortion_lvl))

		_, _, test_loader = utils.load_caltech256(args, dataset_path, indices_path, distortion_lvl)

		for p_tar in p_tar_list:
			print("p_tar: %s"%(p_tar))
			result = run_inference_data(model, test_loader, p_tar, args.n_branches, calib_type, distortion_type, distortion_lvl, device)

			save_result(result, inference_data_path)



def main(args):

	model_path =  os.path.join(config.DIR_NAME, "models", args.dataset_name, args.model_name, "ee_mobilenet_1_branches_id_2.pth")


	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"%s_inference_data_%s_%s_branches_id_%s.csv"%(args.calib_type, args.distortion_type, args.n_branches, args.model_id))

	indices_path = os.path.join(config.DIR_NAME, "indices")
	
	dataset_path = config.dataset_path_dict[args.dataset_name]

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	n_classes = config.nr_class_dict[args.dataset_name]

	#Instantiate the Early-exit DNN model.
	ee_model = ee_nn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, config.dim, device, args.exit_type, args.distribution)

	#Load the trained early-exit DNN model.
	ee_model = ee_model.to(device)

	ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
	ee_model.eval()

	p_tar_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
	distortion_lvl_list = config.distortion_lvl_dict[args.distortion_type]

	extracting_inference_data(ee_model, p_tar_list, distortion_lvl_list, inference_data_path, dataset_path, indices_path, 
		args.calib_type, args.distortion_type, device)
 

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='UCB using MobileNet')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--distortion_type', type=str, default=config.distortion_type, help='Distortion Type.')
	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda ?')
	parser.add_argument('--exit_type', type=str, default=config.exit_type, help='Exit type.')
	parser.add_argument('--distribution', type=str, default=config.distribution, help='Distribution of early exits.')
	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Pretrained ?')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--calib_type', type=str, default="no_calib", help='Calibration Type.')
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, help='Size of train batch.')

	args = parser.parse_args()
	main(args)