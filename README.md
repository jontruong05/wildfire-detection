# Wildfire Detection Using Deep Learning 🔥🌲

This project leverages Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs) to detect forest fires from satellite or drone-captured imagery. The models are trained on a dataset of labeled wildfire and non-wildfire images to predict the presence of fire in unseen images. 

The purpose of this project is to experiment with various neural network architectures and investigate how different design choices impact image predictions. Eight MLPs and nine CNNs were created with the following hidden-layer specifications (architectures that are bolded are the best-performing model of their respective type):

MLPs:
- `mlp_1`: 64
- **`mlp_2`: 128 → 64**
- `mlp_3`: 256 → 64
- `mlp_4`: 512 → 128
- `mlp_5`: 256 → 128 → 64
- `mlp_6`: 1024 → 256 → 64
- `mlp_7`: 512 → 256 → 128 → 64
- `mlp_8`: 1024 → 512 → 256 → 128 → 64

CNNs:
- `cnn_1`: 32 filters (conv size = 3)
- `cnn_2`: 32 filters (conv size = 5)
- `cnn_3`: 32 filters (conv size = 7)
- `cnn_4`: 32 → 64 filters (conv size = 3)
- `cnn_5`: 32 → 64 filters (conv size = 5)
- `cnn_6`: 32 → 64 filters (conv size = 7)
- `cnn_7`: 32 → 64 → 128 filters (conv size = 3)
- `cnn_8`: 32 → 64 → 128 filters (conv size = 5)
- **`cnn_9`: 32 → 64 → 128 filters (conv size = 7)**

## Dataset

The dataset consists of RGB images (250x250) categorized into "fire" and "no_fire" classes, structured in the `ImageFolder` format compatible with PyTorch.

Source: https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data

## Installation and Setup

First, clone this repo: `git clone https://github.com/jontruong05/wildfire-detection.git`

Then, in a terminal, run the following command to install the required dependencies: `pip install torch torchvision numpy matplotlib Pillow`
- If you want to use a GPU, use this command line instead to install PyTorch with CUDA: `pip install torch torchvision numpy matplotlib Pillow --index-url https://download.pytorch.org/whl/cu118`

Before running the notebook, please create two empty folders named `mlp_models` and `cnn_models` in the root directory.

## Running the Code

Run the cells in `wildfire_prediction.ipynb` in a linear fashion. Note that the last two cells may take the longest, since they train, validate, and test each model. 

## Future Work

The best model will be re-implemented in TensorFlow and integrated into a production-ready wildfire detection application.