import torch
from torchvision.models import efficientnet_b2 as EfficientNet
# from torchvision.models import EfficientNet
import numpy as np
import random

class MyVisualization(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		# Perform visualization here
		print("Visualization:", x)
		return x
	
def set_seed(seed):
    """
    Sets the seed for PyTorch, NumPy, and Python's built-in random module.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA devices
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # For reproducibility with CuDNN
    torch.backends.cudnn.benchmark = False  # Deactivate benchmarking for deterministic behavior


def add_module_architecture(new_model, architecture, affected_layers, module):
	if architecture in affected_layers:
		new_model.add_module("prueba", module)
		new_model.add_module("prueba", module)
		return
	
	children = architecture.named_children()

	for name, child in children:
		add_module_architecture(new_model, child, affected_layers, module)

class xCRPEfficientNet(torch.nn.Module):
	def __init__(self):
		super(xCRPEfficientNet, self).__init__()

		efficient_net_vanilla = EfficientNet(weights = "DEFAULT")
		self.features = efficient_net_vanilla.features
		self.avgpool = efficient_net_vanilla.avgpool
		self.classifier = efficient_net_vanilla.classifier

	def forward(self, x):

		# for feat in self.features:
		# 	x = feat(x)
		# 	x *= 0
		
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		
		x = self.classifier(x)
		
		return x



def main():
	# EfficientNetB2 model
	set_seed(0)

	model = xCRPEfficientNet()
	output = model(torch.ones(1, 3, 224, 224))

	print(output)
	# print(model)

	# modified_model = torch.nn.Sequential()
	# i = 0

	# affected_layers = ['Conv2dNormActivation']

	# my_visualization = MyVisualization()
	# add_module_architecture(modified_model, model, affected_layers, my_visualization)

	# Iterate through the original model's modules
	# for name, module in model.named_children():
	# 	# print("Paso por aqui")
	# 	# print(name, module)
	# 	# modified_model.add_module(name, module) 
	# 	my_visualization = MyVisualization()
	# 	modified_model.add_module(f"{i}_custom", my_visualization)
	# 	i += 1

	# add_module_architecture(model, MyVisualization())

	# modified_model([10,10,2020])

if __name__ == '__main__':
	main()
