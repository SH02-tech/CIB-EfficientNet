import torch
from utils.torchmath import ZennitHandler
from collections import defaultdict
import model.metric as module_metric


def evaluate(model, dataloader, device, verbose: bool = True, num_shapes_per_class: int = 5):

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

	# Compute heatmaps for a subsample of 5 images per category

	if verbose:
		print("Computing heatmaps for a subsample...")

	category_samples = defaultdict(list)
	img_names = []

	for idx in range(len(dataloader.dataset)):
		img, label = dataloader.dataset[idx]
		img_name = dataloader.dataset.get_idx_image(idx)
		if len(category_samples[label]) < num_shapes_per_class:
			category_samples[label].append((img, label))
			img_names.append(img_name)
		if all(len(samples) == num_shapes_per_class for samples in category_samples.values()):
			break

	kl_div_per_k = defaultdict(list)
	euclid_per_k = defaultdict(list)
	total_heatmaps = []
	relevances = []

	zennit_handler = ZennitHandler(model)
	layer_name = 'mi_layer.1' if type(model).__name__ == 'xMIEfficientNet' else 'features.8.0'

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

	for k in range(2, 11):
		total_kl_div_dict[k] = float(torch.tensor(kl_div_per_k[k]).mean()) if kl_div_per_k[k] else 0.0
		total_euclid_dict[k] = float(torch.tensor(euclid_per_k[k]).mean()) if euclid_per_k[k] else 0.0

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
		'heatmaps': total_heatmaps,
		'relevances': relevances,
		'image_names': img_names,
	}