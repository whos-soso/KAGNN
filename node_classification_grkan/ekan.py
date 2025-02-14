
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from kat_rational import KAT_Group


class KANLinear(torch.nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            out_features,
            hidden_features=None,
            base_activation=dict(type="KAT", act_init=["gelu", "gelu"]),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = base_activation['act_init'][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = base_activation['act_init'][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x

    


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        #base_activation=torch.nn.SiLU,
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    # hidden_features=hidden_features,
                    # base_activation=base_activation,
                    # bias=bias,
                    # drop=drop,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            x = layer(x)
        torch.cuda.empty_cache()
        return x
