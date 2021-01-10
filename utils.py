"""Utility functions for pruning experiments """
import numpy as np
import torch

def create_dict(model):
    """Creates a dictionary containing all trainable parameters in a given model"""
    module_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_dict[name] = module.weight.detach().numpy()
        elif isinstance(module, torch.nn.Linear):
            module_dict[name] = module.weight.detach().numpy()

def find_difference(first_array, second_array):
    difference = np.subtract(second_array, first_array)
    return difference

def find_smallest(array, n):
    """ Finds the n smallest values in a given array"""
    orig_shape = array.shape
    array_flat = array.flatten()
    mask = np.ones_like(array_flat)

    index_array = np.argpartition(array_flat, n)

    for i in range(n):
        mask[index_array[i]] = 0
    mask.reshape(orig_shape)

def find_largest(array, n):
    """ Finds the n smallest values in a given array"""
    orig_shape = array.shape
    array_flat = array.flatten()
    mask = np.ones_like(array_flat)

    index_array = np.argpartition(array_flat, -n)

    for i in range(n):
        mask[index_array[-i]] = 0
    mask.reshape(orig_shape)
