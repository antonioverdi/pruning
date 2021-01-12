"""Utility functions for pruning experiments """
import numpy as np
import torch
import math

def create_dict(model):
    """Creates a dictionary containing all trainable parameters in a given model"""
    module_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_dict[name] = module.weight.detach().numpy()
        elif isinstance(module, torch.nn.Linear):
            module_dict[name] = module.weight.detach().numpy()

def find_difference(first_array, second_array):
    """ Returns an array that is equal to (second_array - first_array)"""
    difference = np.absolute(np.subtract(second_array, first_array))
    return difference

def find_smallest(array, num):
    """ Finds the n smallest values in a given array"""
    orig_shape = array.shape
    array_flat = array.flatten()
    mask = np.ones_like(array_flat)

    if isinstance(num, float):
        to_prune = math.ceil(num*len(array_flat))
    elif isinstance(num, int):
        to_prune = num

    index_array = np.argpartition(array_flat, to_prune)

    for i in range(to_prune):
        mask[index_array[i]] = 0
    mask.reshape(orig_shape)
    return mask

def find_largest(array, num):
    """ Finds the n smallest values in a given array"""
    orig_shape = array.shape
    array_flat = array.flatten()
    mask = np.ones_like(array_flat)

    index_array = np.argpartition(array_flat, -num)

    for i in range(num):
        mask[index_array[-i]] = 0
    mask.reshape(orig_shape)
    return mask
