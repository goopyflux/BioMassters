"""Utility functions used by the module."""

import torch


INPUT_IMAGE_DIMS = (3, 256, 256)
OUTPUT_IMAGE_DIMS = (256, 256)

def get_random_image():
    random_image = torch.rand(INPUT_IMAGE_DIMS)

def make_output_tensor(image):
    image = image
    return torch.rand(OUTPUT_IMAGE_DIMS)