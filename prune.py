""" Pruning experiments with pytorch"""
import torch
import utils

def smallest_delta(model1, model2, amount):
    """Prunes the weights that have changed the least between model1 and model2 """
    for name1, module1 in model1.named_modules():
        for name2, module2 in model2.named_modules():
            if name1 == name2:
                if isinstance(module1, torch.nn.Conv2d):
                    array1 = module1.weight.detach().numpy()
                    array2 = module2.weight.detach().numpy()
                    difference = utils.find_difference(array1, array2)
                    mask = utils.find_smallest(difference, amount)
                    module2.weight = utils.apply_mask(mask, array2)

def greatest_delta(model1, model2, amount):
    """Prunes the weights that have changed the most between model1 and model2 """
    for name1, module1 in model1.named_modules():
        for name2, module2 in model2.named_modules():
            if name1 == name2:
                if isinstance(module1, torch.nn.Conv2d):
                    array1 = module1.weight.detach().numpy()
                    array2 = module2.weight.detach().numpy()
                    difference = utils.find_difference(array1, array2)
                    mask = utils.find_largest(difference, amount)
                    module2.weight = utils.apply_mask(mask, array2)
