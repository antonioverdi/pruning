"""Utility functions for pruning experiments """
import math
import numpy as np
import torch
import torch.nn as nn

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

def find_smallest(array, amount, already_pruned):
	""" Finds the n smallest values in a given array"""
	orig_shape = array.shape
	array_flat = array.flatten()
	mask = np.ones_like(array_flat)

	if isinstance(amount, float):
		amount = math.ceil(amount*len(array_flat))

	already_pruned = (already_pruned-1)*amount
	index_array = np.argpartition(array_flat, amount+already_pruned)

	for i in range(already_pruned, amount):
		mask[index_array[i]] = 0

	mask = mask.reshape(orig_shape)
	already_pruned += amount
	return mask

def find_greatest(array, amount):
	""" Finds the n greatest values in a given array"""
	orig_shape = array.shape
	array_flat = array.flatten()
	mask = np.ones_like(array_flat)

	if isinstance(amount, float):
		amount = math.ceil(amount*len(array_flat))

	index_array = np.argpartition(array_flat, -amount)

	for i in range(amount):
		mask[index_array[-i]] = 0
	mask = mask.reshape(orig_shape)
	return mask

def apply_mask(mask, array):
	return nn.Parameter(torch.from_numpy(np.multiply(mask, array)).float().to("cuda:0")) 

def calculate_sparsity(model):
	sparsities = []
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Conv2d):
			nonzero = torch.count_nonzero(module.weight.detach()).item()
			numel = len(module.weight.detach().numpy().flatten())
			sparsities.append(100*(1-(nonzero/numel)))
		elif isinstance(module, torch.nn.Linear):
			nonzero = torch.count_nonzero(module.weight.detach()).item()
			numel = len(module.weight.detach().numpy().flatten())
			sparsities.append(100*(1-(nonzero/numel)))
	return sparsities
	