import torch
import torch.nn.functional as F

class EdgeNoiseGenerator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim
        self.mlp_feat_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )        
        self.mlp_edge_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        feat = self.mlp_feat_model(x)
        emb_src = feat[src]
        emb_dst = feat[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits
    
    def apply_gumbel_softmax(self, probs, device):
        temperature = 1.0
        bias = 0.0 + 0.0001  
        eps = (bias - (1 - bias)) * torch.rand(probs.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + probs) / temperature  
        noise_adj = torch.sigmoid(gate_inputs).squeeze() 
        return noise_adj
    
class FeatNoiseGenerator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_mean = torch.nn.Sequential(torch.nn.Linear(input_dim, self.hidden_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(self.hidden_dim, self.input_dim))
        self.fc_variance = torch.nn.Sequential(torch.nn.Linear(input_dim, self.hidden_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(self.hidden_dim, self.input_dim))


    def freeze(self, flag=True):
        for p in self.parameters():
            p.requires_grad = not flag

    def forward(self, feat):
        variance = self.fc_variance(feat).abs()
        mu = self.fc_mean(feat)
        return mu, variance

    def sampling(self, mu, variance):        
        noise = torch.randn(mu.shape).to(variance.device)
        
        noise = variance * noise + mu
        
        return noise
    