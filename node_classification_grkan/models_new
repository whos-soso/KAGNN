from ekan import KAN as eKAN,KANLinear
from fastkan import FastKAN,FastKANLayer
import torch

import torch.nn as nn
from torch_geometric.nn import GINConv, GCNConv, GATConv
from kat_rational import KAT_Group


def make_mlp(num_features, hidden_dim, out_dim, hidden_layers):
    if hidden_layers>=2:
        list_hidden = [nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU())]
        for _ in range(hidden_layers-2):
            list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        list_hidden.append(nn.Sequential(nn.Linear(hidden_dim, out_dim, nn.ReLU())))
    else:
        list_hidden = [nn.Sequential(nn.Linear(num_features, out_dim), nn.ReLU())]
    mlp = nn.Sequential(*list_hidden)
    return(mlp)

class KANLayer(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            out_features,
            hidden_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
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

class KAGCNConv(GCNConv):
    def __init__(self, in_feat:int,
                 out_feat:int):
        super(KAGCNConv, self).__init__(in_feat, out_feat)
        self.lin = KANLayer(in_feat, out_feat)

class KAGATConv(GATConv):
    def __init__(self, in_feat:int,
                 out_feat:int,
                 heads:int):
        super(KAGATConv, self).__init__(in_feat, out_feat, heads)
        self.lin = KANLayer(in_feat, out_feat*heads)


class GNN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 hidden_layers:int=2,
                 dropout:float=0.,
                 heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if conv_type!='gat':
            heads = 1
        torch.nn.BatchNorm1d(hidden_channels)
        for i in range(mp_layers):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(num_features, hidden_channels))
                elif conv_type == "gat":
                    self.convs.append(GATConv(num_features, hidden_channels, heads))
                elif conv_type == "gin":
                    self.convs.append(GINConv(make_mlp(num_features, hidden_channels, hidden_channels, hidden_layers)))
                else:
                    raise ValueError("unknown conv_type")
            else:
                if conv_type == "gcn":
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                elif conv_type == "gat":
                    self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads))
                elif conv_type == "gin":
                    self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels, hidden_channels, hidden_layers)))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        self.skip = skip
        dim_out_message_passing = num_features+(mp_layers)*hidden_channels if skip else hidden_channels
        if conv_type == "gat":
            dim_out_message_passing = num_features+(mp_layers)*hidden_channels*heads if skip else hidden_channels*heads
        self.lay_out = torch.nn.Linear(dim_out_message_passing, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        if self.skip:
            l.append(x)
        for conv,bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.dropout(x)
            if self.skip:
                l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.lay_out(x)
        return x

class GKAN_Nodes(torch.nn.Module):
    def __init__(self,  conv_type :str,
                 mp_layers:int,
                 num_features:int,
                 hidden_channels:int,
                 num_classes:int,
                 skip:bool = True,
                 grid_size:int = 4,
                 spline_order:int = 3,
                 hidden_layers:int=2,
                 dropout:float=0.,
                 heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if conv_type!='gat':
            heads = 1
        for i in range(mp_layers):
            if i ==0:
                if conv_type == "gcn":
                    self.convs.append(KAGCNConv(num_features, hidden_channels, grid_size, spline_order))
                elif conv_type == "gat":
                    self.convs.append(KAGATConv(num_features, hidden_channels, heads, grid_size, spline_order))
                elif conv_type == "gin":
                    self.convs.append(GIKANLayer(num_features, hidden_channels, grid_size, spline_order, hidden_channels, hidden_layers))
                else:
                    raise ValueError("unknown conv_type")
            else:
                if conv_type == "gcn":
                    self.convs.append(KAGCNConv(hidden_channels, hidden_channels, grid_size, spline_order))
                elif conv_type == "gat":
                    self.convs.append(KAGATConv(hidden_channels*heads, hidden_channels, heads, grid_size, spline_order))
                else:
                    self.convs.append(GIKANLayer(hidden_channels, hidden_channels, grid_size, spline_order, hidden_channels, hidden_layers))
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        self.skip = skip
        dim_out_message_passing = num_features+mp_layers*hidden_channels if skip else hidden_channels
        if conv_type == "gat":
            dim_out_message_passing = num_features+mp_layers*hidden_channels*heads if skip else hidden_channels*heads
        self.lay_out = KANLinear(dim_out_message_passing, num_classes, grid_size=grid_size, spline_order=spline_order)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        l = []
        l.append(x)
        for conv,bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.dropout(x)
            l.append(x)
        if self.skip:
            x = torch.cat(l, dim=1)
        x = self.lay_out(x)
        return x

