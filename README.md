# CIB-EfficientNet

In this repository we implement CIB-EfficientNet (Concise Information in 
Bottleneck EfficientNet). CIB is method used to increase explainability
presented in this Bachelor's Thesis (TBA). 

## Project Structure

This repo presents this structure:

```
model/
	loss.py
	metric.py
	model.py
notebooks/
	attributions.ipynb
	plotter.ipynb
data_loader/
	data_loaders.py
train.py
test.py
config.json
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SH02-tech/CIB-EfficientNet.git
cd CIB-EfficientNet
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate fcrp
```

3. Verify the installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

### Training

To train the model, use the training script with a configuration file:

```bash
python train.py --config config_prostate.json
```

Available configuration files:
- `config_prostate.json` - Full prostate dataset configuration
- `config_prostate_reduced.json` - Reduced prostate dataset configuration
- `config_prostate_xmi.json` - XMI prostate dataset configuration
- `config_prostate_xmi_reduced.json` - Reduced XMI prostate dataset configuration

### Testing

To evaluate the trained model:

```bash
python test.py --config config_prostate.json --resume path/to/checkpoint.pth
```

### Notebooks

The project includes several Jupyter notebooks for analysis and visualization:

- `attributions.ipynb` - Attribution analysis and visualization
- `plottter.ipynb` - Plotting utilities and visualizations
- `test_results_helper.ipynb` - Helper functions for test result analysis
- `test_results.ipynb` - Test results analysis and visualization

## Configuration

The configuration files contain parameters for:
- Model architecture settings
- Training hyperparameters
- Dataset paths and preprocessing options
- Logging and checkpoint settings

## Data Structure

The project expects data to be organized according to the splits defined in `data_split/`:
- `train.txt` / `train_reduced.txt` - Training set file lists
- `val.txt` / `val_reduced.txt` - Validation set file lists
- `test.txt` / `test_reduced.txt` - Test set file lists
- `classes.txt` - Class labels definition

## Scripts

Utility scripts are available in the `scripts/` directory:
- `train.sh` - Training script wrapper
- `test.sh` - Testing script wrapper
- `create_sets_files.sh` - Data split creation utility
- `tensorboard_last_n.sh` - TensorBoard logging utility

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
