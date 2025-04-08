import copy
import torch

def compute_sparsity(state_dict):
    total = 0
    zeros = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.numel()
            zeros += (v == 0).sum().item()
    return zeros / total if total > 0 else 0.0

def prune_model(state_dict, pruning_rate=0.5): # layer-wise magnitude pruning
    """
    Simple layer-wise magnitude pruning:
    - pruning_rate: 0.5 -> make half of model parameters to 0
    """
    pruned = copy.deepcopy(state_dict)

    for k, v in pruned.items():
        if 'weight' in k and isinstance(v, torch.Tensor):
            num_elements = v.numel()
            k_val = int(pruning_rate * num_elements)

            if k_val == 0:
                continue

            flat = v.view(-1)
            threshold = torch.topk(flat.abs(), k_val, largest=False).values.max()
            mask = (flat.abs() >= threshold).float()
            pruned[k] = (flat * mask).view_as(v)

    return pruned
