import torch
from utils.torchmath import ZennitHandler
from collections import defaultdict
import model.metric as module_metric

def evaluate(model, dataloader, device, verbose: bool = True, num_shapes_per_class: int = 5):
	"""
	Evaluate the model on the given dataloader.

	Args:
		model: The model to evaluate.
		dataloader: The dataloader containing the dataset.
		device: The device to run the evaluation on.
		verbose: If True, print evaluation progress.
		num_shapes_per_class: Number of shapes to sample per class for heatmap
		computation.

	Returns:
		A dictionary containing the evaluation metrics:
			- 'accuracy': Overall accuracy of the model.
			- 'per_class_accuracy': Dictionary with per-class accuracy.
			- 'loss': Total loss over the dataset.
			- 'kl_divergence': Dictionary with KL divergence for each k.
			- 'euclidean_distance': Dictionary with Euclidean distance for each k.
			- 'mean_euclidean_distance': Mean Euclidean distance across all k.
			- 'heatmaps': List of heatmaps computed for the sampled images.
			- 'relevances': List of relevance scores for the sampled images.
			- 'images_info': List of tuples containing image names and images.

		If the model is an xMI-EfficientNet, it will also compute heatmaps
		from the bottleneck layer and add them to the dictionary with keys
		suffixed with '_bottleneck'.
	"""
	dict = evaluate_individual(
		model, 
		dataloader, 
		device, 
		verbose=verbose, 
		num_shapes_per_class=num_shapes_per_class,
		bottleneck_layer=False
	)

	if 'xMIEfficient' in type(model).__name__:
		dict_bottleneck = evaluate_individual(
			model, 
			dataloader, 
			device, 
			verbose=verbose, 
			num_shapes_per_class=num_shapes_per_class,
			bottleneck_layer=True
		)
	
		for key in dict_bottleneck:
			if key in dict:
				dict[key + '_bottleneck'] = dict_bottleneck[key]
			else:
				dict[key] = dict_bottleneck[key]

	return dict

def evaluate_individual(model, dataloader, device, verbose: bool = True, num_shapes_per_class: int = 5, bottleneck_layer: bool = False):

	total_loss = 0.0
	total_accuracy = 0.0
	total_kl_div_dict = {}
	total_euclid_dict = {}
	total_heatmaps = []
	
	n_samples = len(dataloader.dataset)

	# For per-class accuracy
	class_correct = defaultdict(int)
	class_total = defaultdict(int)

	if verbose:
		print("Evaluating model on test set...")

	with torch.no_grad():
		for i, (data, target) in enumerate(dataloader):
			data = data.to(device)
			target = target.to(device)

			if type(model).__name__ == 'xMIEfficientNet':
				output, mi_layer_weights, features = model(data, output_features=True)
			else:
				output = model(data)

			batch_size = data.shape[0]
			total_accuracy += module_metric.accuracy(output, target) * batch_size

			# Per-class accuracy
			_, preds = torch.max(output, 1)
			for label in torch.unique(target):
				label_mask = (target == label)
				class_correct[int(label)] += (preds[label_mask] == label).sum().item()
				class_total[int(label)] += label_mask.sum().item()

	category_samples = {label: [] for label in range(14)}
	images_info = []

	for idx in range(len(dataloader.dataset)):
		img, label = dataloader.dataset[idx]
		img_name = dataloader.dataset.get_idx_image(idx)
		if len(category_samples[label]) < num_shapes_per_class:
			category_samples[label].append((img, label))
			images_info.append((img_name, img))
		if all(len(samples) == num_shapes_per_class for samples in category_samples.values()):
			break

	kl_div_per_k = defaultdict(list)
	euclid_per_k = defaultdict(list)
	total_heatmaps = []
	relevances = []

	zennit_handler = ZennitHandler(model)
	layer_name = 'mi_layer.1' if bottleneck_layer else 'features.8.0'

	# Compute heatmaps for a subsample of 5 images per category

	if verbose:
		print("Computing heatmaps for a subsample...")


	for label, samples in category_samples.items():
		for img, target in samples:
			img = img.to(device)
			attr, rel = zennit_handler.get_attr_top(img, layer_name, target, k=15)
			heatmap = attr.heatmap
			norm_heatmap = (heatmap - heatmap.mean()) / heatmap.std()
			total_heatmaps.append(heatmap)
			relevances.append(rel)
			for k in range(2, 11):
				kl = module_metric.kl_divergence_hm(norm_heatmap, k)
				euclid = module_metric.euclidean_distance_hm(norm_heatmap, k)
				kl_div_per_k[k].append(kl)
				euclid_per_k[k].append(euclid)

	mean_euclid = 0.0
	euclid_values = []
	for k in range(2, 11):
		if euclid_per_k[k]:
			mean_k = float(torch.tensor(euclid_per_k[k]).mean())
			total_euclid_dict[k] = mean_k
			euclid_values.extend(euclid_per_k[k])
		else:
			total_euclid_dict[k] = 0.0
		if kl_div_per_k[k]:
			total_kl_div_dict[k] = float(torch.tensor(kl_div_per_k[k]).mean())
		else:
			total_kl_div_dict[k] = 0.0

	if euclid_values:
		mean_euclid = float(torch.tensor(euclid_values).mean())
	else:
		mean_euclid = 0.0

	# Compute per-class accuracy
	per_class_accuracy = {}
	for label in class_total:
		if class_total[label] > 0:
			per_class_accuracy[label] = class_correct[label] / class_total[label]
		else:
			per_class_accuracy[label] = 0.0

	return {
		'accuracy': total_accuracy / n_samples,
		'per_class_accuracy': per_class_accuracy,
		'loss': total_loss,
		'kl_divergence': total_kl_div_dict,
		'euclidean_distance': total_euclid_dict,
		'mean_euclidean_distance': mean_euclid,
		'heatmaps': total_heatmaps,
		'relevances': relevances,
		'images_info': images_info
	}