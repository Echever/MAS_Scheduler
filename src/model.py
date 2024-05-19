import torch
import torch
import torch.nn as nn
from torch_geometric.nn import  Linear, to_hetero
from torch_geometric.data import HeteroData
import torch_scatter
from transformer_conv import TransformerConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels,  num_layers = 2, heads = 3):
        super().__init__()
        self.lin1 = Linear(-1, 8)
        self.s = torch.nn.Softmax(dim=0)
        self.activation = nn.Tanh()  
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = TransformerConv(-1, hidden_channels, edge_dim=4, heads = heads)
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr_dict):
        x = self.lin1(x)
        x = self.activation(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr_dict)
            x = self.activation(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, num_layers = 2, heads = 3):
        super().__init__()
        self.gnn = GAT(hidden_channels, num_layers=num_layers, heads=heads)
        self.gnn = to_hetero(self.gnn, metadata=metadata, aggr='mean')
        self.lin3 = Linear(-1, 1)
        self.lin4 = Linear(-1, hidden_channels)
        self.activation = nn.Tanh()

    def forward(self, data: HeteroData):
        gres = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        x_src, x_dst = gres['machine'][data.edge_index_dict[('machine','exec','job')][0]], gres['job'][data.edge_index_dict['machine','exec','job'][1]]
        mask = data['machine', 'exec', 'job'].final_consideration
        res = torch.cat([x_src,  data.edge_attr_dict[('machine','exec','job')], x_dst], dim=-1)[mask]

        res = self.lin4(res)
        res = self.activation(res)
        res = self.lin3(res)

        return res
    