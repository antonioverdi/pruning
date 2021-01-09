""" Pruning experiments with pytorch"""
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_net(net):
    """ Utility function for pruning network"""
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return None
    newnet = copy.copy(net)
    modules_list = []

    for name, module in newnet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            modules_list += [(module,'weight'),(module,'bias')]
        if isinstance(module, torch.nn.Linear):
            modules_list += [(module,'weight'),(module,'bias')]

    prune.global_unstructured(
        modules_list,
        pruning_method=prune.L1Unstructured,
        amount=0.2,)
    return newnet
