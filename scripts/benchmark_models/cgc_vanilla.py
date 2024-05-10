from typing import Dict
from argparse import Namespace

import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data import Batch

from gnn.mace import get_edge_vectors_and_lengths

class CGCLayer(torch.nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, reduction: str = 'sum') -> None:
        super().__init__()
        
        self.num_hid_dim = 2*node_dim+edge_dim
        self.fc_values = torch.nn.Linear(self.num_hid_dim, node_dim)
        self.fc_multip = torch.nn.Linear(self.num_hid_dim, node_dim)

        self.reduction = reduction

    def forward(self, x: Tensor, edge_index: Tensor, edge_ft: Tensor):
        sender, receiver = edge_index
        c = torch.cat([x[sender], x[receiver], edge_ft], dim=1) # [num_edges, num_hid_dim]
        msg = torch.nn.functional.softplus(self.fc_values(c))*torch.sigmoid(self.fc_multip(c))
        return scatter(msg, receiver, dim=0, dim_size=x.shape[0], reduce=self.reduction)


class CrystGraphConv(torch.nn.Module):
    def __init__(self, params: Namespace) -> "CrystGraphConv":
        super().__init__()
        self.params = params

        hidden_dim = params.hidden_irreps
        self.node_ft_embedding = torch.nn.Linear(in_features=3, out_features=hidden_dim)
        self.edge_ft_embedding = torch.nn.Linear(in_features=5, out_features=hidden_dim)
        
        self.cgc_layers = torch.nn.ModuleList(
            [CGCLayer(hidden_dim, hidden_dim, params.interaction_reduction) for i in range(params.message_passes)]
        )

        self.global_reduction = params.global_reduction

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 32),
            torch.nn.Softplus(),
            torch.nn.Linear(32, 21)
        )

    def forward(self, batch: Batch) -> Dict:
        
        num_graphs = batch.num_graphs
        edge_index = batch.edge_index
        node_ft = batch.positions # [num_nodes, 3]
        edge_radius = batch.edge_attr # [num_edges, 1]

        node_ft = self.node_ft_embedding(node_ft) # [num_nodes, hidden_dim]

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch.positions, edge_index=edge_index, shifts=batch.shifts, normalize=True
        )
        edge_ft = torch.cat([vectors, lengths, edge_radius], dim=1) # [num_edges, 5]
        edge_ft = self.edge_ft_embedding(edge_ft) # [num_edges, hidden_dim]

        for i_mp in range(self.params.message_passes):
            node_ft = node_ft + self.cgc_layers[i_mp](node_ft, edge_index, edge_ft)
        
        graph_ft = scatter(node_ft, batch.batch, dim=0, dim_size=num_graphs, reduce=self.global_reduction) # [num_graphs, hidden_dim]

        stiffness = self.mlp(graph_ft) # [num_graphs, 21]

        return {'stiffness': stiffness}