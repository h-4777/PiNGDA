import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, classes_num):
        super().__init__()
        self.fc = nn.Linear(feat_dim, classes_num)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        out = self.fc(x)
        return out

class WGINConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(WGINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)


    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class TUEncoder(torch.nn.Module):
    def __init__(self, num_dataset_features, emb_dim, num_gc_layers, drop_ratio, pooling_type, is_infograph=False):
        super(TUEncoder, self).__init__()

        self.pooling_type = pooling_type
        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.is_infograph = is_infograph

        self.out_node_dim = self.emb_dim
        if self.pooling_type == "standard":
            self.out_graph_dim = self.emb_dim
        elif self.pooling_type == "layerwise":
            self.out_graph_dim = self.emb_dim * self.num_gc_layers
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            else:
                nn = Sequential(Linear(num_dataset_features, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
            conv = WGINConv(nn)
            bn = torch.nn.BatchNorm1d(emb_dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, batch, x, edge_index, edge_weight=None):
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)
        if self.pooling_type == "standard":
            xpool = global_add_pool(x, batch)
            return xpool, x

        elif self.pooling_type == "layerwise":
            xpool = [global_add_pool(x, batch) for x in xs]
            xpool = torch.cat(xpool, 1)
            if self.is_infograph:
                return xpool, torch.cat(xs, 1)
            else:
                return xpool, x
        else:
            raise NotImplementedError

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(batch, x, edge_index, edge_weight)

                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    
class Model(torch.nn.Module):
    def __init__(self, encoder, proj_hidden_dim=300):
        super(Model, self).__init__()

        self.encoder = encoder
        self.input_proj_dim = self.encoder.out_graph_dim

        self.proj_head = Sequential(Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
                                       Linear(proj_hidden_dim, proj_hidden_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):

        z, node_emb = self.encoder(batch, x, edge_index, edge_attr, edge_weight)

        z = self.proj_head(z)
        return z, node_emb
    
    def cal_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim = torch.mm(z1, z2.t())
        return torch.exp(sim / 0.3)
    
    def cal_loss(self, z1, z2, batch_compute=False, batch_size=256):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        if not batch_compute:
            indices = torch.arange(0, num_nodes).to(device)
        else:
            indices = torch.randperm(num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            if not batch_compute:
                refl_sim = self.cal_sim(z1[mask], z1)
                between_sim = self.cal_sim(z1[mask], z2)
                losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
            else:
                refl_sim = self.cal_sim(z1[mask], z1[mask])
                between_sim = self.cal_sim(z1[mask], z2[mask])
                losses.append(-torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())))
        return torch.cat(losses)  