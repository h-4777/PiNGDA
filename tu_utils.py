import numpy as np
import random
import torch
from torch.utils.data import random_split

def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def drop_edges(edge_index, p):
    sel_mask = torch.bernoulli(torch.full((edge_index.shape[1],), p)).to(torch.bool)
    
    return edge_index[:, sel_mask]

def to_device(device, *args):
    if not isinstance(device, torch.device):
        raise Exception('The first parameter is not a valid torch device instance')
    for arg in args:
        arg.to(device)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)


def get_Laplacian_from_adj(adjacency):
    adj = adjacency + torch.eye(adjacency.shape[0]).to(adjacency)
    degree = torch.sum(adj, dim=1).pow(-0.5)
    return (adj * degree).t() * degree

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask


def initialize_edge_weight(data):
	data.edge_weight = torch.ones(data.edge_index.shape[1], dtype=torch.float)
	return data

def initialize_node_features(data):
	num_nodes = int(data.edge_index.max()) + 1
	data.x = torch.ones((num_nodes, 1))
	return data

def set_tu_dataset_y_shape(data):
	num_tasks = 1
	data.y = data.y.unsqueeze(num_tasks)
	return data