"""DeepONet's fully connected block (from deeponet.py, structurally identical to
DeepOHeat's src.modules.FCBlock)."""
import torch
import torch.nn as nn


class FCBlock(nn.Module):
    """Structurally identical to DeepOHeat's src.modules.FCBlock:
    in -> [hidden] x num_hidden -> out, silu."""
    def __init__(self, in_features, hidden_features, out_features, num_hidden_layers):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_features), nn.SiLU()]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_features, hidden_features), nn.SiLU()]
        layers += [nn.Linear(hidden_features, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
