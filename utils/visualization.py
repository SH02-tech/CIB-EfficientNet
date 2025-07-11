import re
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utils.torchmath import denormalize

def plot_train_val(log_path, save_file=None, y_label='Loss', title='Curva de aprendizaje', subloss_type=None, fontsize=20, smooth=False, sigma=1):
	# Regex to match lines like: "epoch          : 1"
	epoch_re = re.compile(r"epoch\s*:\s*(\d+)")
	# Regex to match lines like: "loss           : 1.4490584661794264"
	if subloss_type is None:
		loss_re = re.compile(r"loss\s*:\s*([0-9\.eE+-]+)")
		val_loss_re = re.compile(r"val_loss\s*:\s*([0-9\.eE+-]+)")
	else:
		# Match e.g. "loss/nll_loss  : 1.2839036902715995"
		loss_re = re.compile(r"loss/{0}\s*:\s*([0-9\.eE+-]+)".format(re.escape(subloss_type)))
		val_loss_re = re.compile(r"val_loss/{0}\s*:\s*([0-9\.eE+-]+)".format(re.escape(subloss_type)))

	epochs = []
	losses = []
	val_losses = []
	with open(log_path, "r") as f:
		lines = f.readlines()

	i = 0
	while i < len(lines):
		line = lines[i]
		epoch_match = epoch_re.search(line)
		if epoch_match:
			epoch = int(epoch_match.group(1))
			loss = None
			val_loss = None
			# Search for loss and val_loss in the next 20 lines
			for j in range(1, 20):
				if i + j < len(lines):
					l2 = lines[i + j]
					if loss is None:
						m = loss_re.search(l2)
						if m:
							loss = float(m.group(1))
					if val_loss is None:
						m = val_loss_re.search(l2)
						if m:
							val_loss = float(m.group(1))
				if loss is not None and val_loss is not None:
					break
			if loss is not None and val_loss is not None:
				epochs.append(epoch)
				losses.append(loss)
				val_losses.append(val_loss)
		i += 1

	if not epochs:
		print("No epochs found in log.")
		return

	plt.figure(figsize=(10, 6))
	
	if smooth:
		# Smooth with gaussian filter
		losses_smooth = gaussian_filter1d(losses, sigma=sigma)
		val_losses_smooth = gaussian_filter1d(val_losses, sigma=sigma)
		
		plt.plot(epochs, losses, 'o-', alpha=0.3, label="Loss (raw)", color='tab:blue')
		plt.plot(epochs, losses_smooth, '-', linewidth=2, label="Loss (smoothed)", color='tab:blue')
		plt.plot(epochs, val_losses, 'o-', alpha=0.3, label="Val Loss (raw)", color='tab:orange')
		plt.plot(epochs, val_losses_smooth, '-', linewidth=2, label="Val Loss (smoothed)", color='tab:orange')
	else:
		# Plot only raw data
		plt.plot(epochs, losses, 'o-', label="Loss", color='tab:blue')
		plt.plot(epochs, val_losses, 'o-', label="Val Loss", color='tab:orange')
	plt.xlabel("Época", fontsize=fontsize)
	plt.ylabel(y_label, fontsize=fontsize)
	plt.title(title, fontsize=fontsize)
	plt.legend(fontsize=fontsize)
	# plt.grid(True)
	# Set x-ticks every 5 epochs starting from 5 for clarity
	xtick_epochs = [e for e in epochs if e % 5 == 0 and e >= 5]
	plt.xticks(xtick_epochs, fontsize=fontsize)
	plt.xlim(left=0)
	plt.tight_layout()
	if save_file:
		plt.savefig(save_file, format='pdf')
		print(f"Plot saved to {save_file}")
	plt.show()

def show_batch_denormalized(list_tensors, mean, std, titles=None, num_cols=4, figsize=None):
    """
    Visualizes a batch of DENORMALIZED image tensors in a grid.
    Args:
        list_tensors (torch.Tensor): List of image tensors (C, H, W), ASSUMED NORMALIZED, with corresponding label
        mean (list or tuple): Mean values used for normalization.
        std (list or tuple): Standard deviation values used for normalization.
        titles (list of str, optional): List of titles for each image.
        num_cols (int): Number of columns in the grid.
        figsize (tuple): Figure size (width, height).
    """
    batch_size = len(list_tensors)
    num_rows = math.ceil(batch_size / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (img_tensor, label) in enumerate(list_tensors):
        ax = axes[i]
        
        # Denormalize the image tensor
        denormalized_img_tensor = denormalize(img_tensor, mean, std)

        # Convert to numpy and handle channel order
        if denormalized_img_tensor.shape[0] == 1: # Grayscale
            img_np = denormalized_img_tensor.squeeze().cpu().numpy()
        else: # RGB
            img_np = denormalized_img_tensor.permute(1, 2, 0).cpu().numpy()
            
        # Clip values to [0, 1] in case of floating point inaccuracies
        img_np = np.clip(img_np, 0, 1)
            
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title("sample")
        if titles and i < len(titles):
            ax.set_title(titles[i])
        else:
            ax.set_title(f"Label: {label}")
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def show_hm(hm, img, save_path=None):
	"""
	Displays a heatmap overlay on the original image. Save the figure if 
	save_path is provided.
	Args:
		hm (torch.Tensor): The heatmap tensor (C, H, W).
		img (torch.Tensor): The original image tensor (C, H, W).
		save_path (str, optional): Path to save the figure.
	"""
	norm_hm = (hm - hm.min()) / (hm.max() - hm.min())
	threshold = torch.quantile(norm_hm, 0.9).item()
	norm_hm = norm_hm.cpu().numpy()
	norm_hm[norm_hm < threshold] = 0 # clearer visualization

	# Show heatmap
	plt.figure(figsize=(6,6))
	plt.imshow(img.permute(1, 2, 0).cpu().numpy(), alpha=0.9)
	plt.imshow(norm_hm, cmap='hot', vmin=0, alpha=0.5)
	plt.axis('off')
	if save_path:
		plt.savefig(save_path)
		plt.close()
	else:
		plt.show()