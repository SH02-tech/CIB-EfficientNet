import re
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from utils.torchmath import denormalize

def plot_train_val(log_path):

	epoch_re = re.compile(r"epoch\s*:\s*(\d+)")
	loss_re = re.compile(r"loss\s*:\s*([0-9\.eE+-]+)")
	val_loss_re = re.compile(r"val_loss\s*:\s*([0-9\.eE+-]+)")

	epochs = []
	losses = []
	val_losses = []

	with open(log_path, "r") as f:
		lines = f.readlines()

	for i in range(len(lines)):
		line = lines[i]
		epoch_match = epoch_re.search(line)
		if epoch_match:
			epoch = int(epoch_match.group(1))
			# Buscar las siguientes líneas para loss y val_loss
			loss = None
			val_loss = None
			for j in range(1, 10):  # Buscar en las siguientes 10 líneas
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

	# Suavizado con filtro gaussiano
	losses_smooth = gaussian_filter1d(losses, sigma=1)
	val_losses_smooth = gaussian_filter1d(val_losses, sigma=1)

	plt.figure(figsize=(10,6))
	plt.plot(epochs, losses, 'o-', alpha=0.3, label="Loss (raw)", color='tab:blue')
	plt.plot(epochs, losses_smooth, '-', linewidth=2, label="Loss (smoothed)", color='tab:blue')
	plt.plot(epochs, val_losses, 'o-', alpha=0.3, label="Val Loss (raw)", color='tab:orange')
	plt.plot(epochs, val_losses_smooth, '-', linewidth=2, label="Val Loss (smoothed)", color='tab:orange')
	plt.xlabel("Época")
	plt.ylabel("Loss")
	plt.title("Entrenamiento/Validación")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
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
