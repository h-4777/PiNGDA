import torch
import random

def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def drop_edges(edge_index, p):
    sel_mask = torch.bernoulli(torch.full((edge_index.shape[1],), p)).to(torch.bool)
    
    return edge_index[:, sel_mask]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True