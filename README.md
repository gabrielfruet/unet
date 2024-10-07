# UNET Implementation using Keras and TensorFlow

This repository contains an implementation of the UNET architecture using Keras
and TensorFlow. UNET is a popular convolutional neural network architecture
primarily used for image segmentation tasks, where the goal is to predict a
label for each pixel in an image.

## Overview

UNET was originally proposed by Olaf Ronneberger et al. in their paper titled
["U-Net: Convolutional Networks for Biomedical Image
Segmentation"](https://arxiv.org/abs/1505.04597). It has since become a
powerful and widely used architecture for various image segmentation tasks, not
just in biomedical imaging, but also in fields such as satellite image
analysis, self-driving cars, and more.

This implementation follows the general structure of the UNET, comprising a
contracting path (encoder) and an expansive path (decoder), with skip
connections between corresponding layers in the encoder and decoder.

## Requirements

To install the required dependencies, you can use:

```bash pip install -r requirements.txt ```

## Usage

1. Clone the repository:

    ```bash git clone <repository-url> cd <repository-directory> ```

2. Customize layers or add new functionalities by editing `src/layers.py`.

3. Train the UNET model using your dataset:

    Instantiate the model with:

    ```python
    from src import UNet

    model = UNet(input_size=(HEIGHT, WIDTH, CHANNELS), n_filters=N, n_classes=M)
    ```

    Your dataset should be on the format `(img, mask, weight_map)` for training.

    Your loss function should support the `weight_map` as the argument `sample_weight`.

    ```python
    from src import weight_map_focal_loss

    model.compile(..., loss=weight_map_focal_loss, ...)
    model.fit(...)
    ```


## Citation

If you use this implementation in your work, please consider citing the
original paper:

``` 
@article{
    ronneberger2015u, 
    title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
    author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas}, 
    journal={arXiv preprint arXiv:1505.04597},
    year={2015} 
} 
```

## Acknowledgments

This implementation is based on the original UNET paper and leverages
TensorFlow and Keras for model construction and training.
