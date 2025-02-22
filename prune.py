""" Pruning experiments with pytorch"""
import torch
import utils

def prune_smallest(model1, model2, amount, already_pruned):
    """Prunes the weights that have changed the least between model1 and model2 """
    for name1, module1 in model1.named_modules():
        for name2, module2 in model2.named_modules():
            if name1 == name2:
                if isinstance(module1, torch.nn.Conv2d):
                    array1 = module1.weight.detach().cpu().numpy()
                    array2 = module2.weight.detach().cpu().numpy()
                    difference = utils.find_difference(array1, array2)
                    mask = utils.find_smallest(difference, amount, already_pruned)
                    module2.weight = utils.apply_mask(mask, array2)

    return model2

def prune_greatest(model1, model2, amount):
    """Prunes the weights that have changed the most between model1 and model2 """
    for name1, module1 in model1.named_modules():
        for name2, module2 in model2.named_modules():
            if name1 == name2:
                if isinstance(module1, torch.nn.Conv2d):
                    array1 = module1.weight.detach().cpu().numpy()
                    array2 = module2.weight.detach().cpu().numpy()
                    difference = utils.find_difference(array1, array2)
                    mask = utils.find_greatest(difference, amount)
                    module2.weight = utils.apply_mask(mask, array2)

    return model2
