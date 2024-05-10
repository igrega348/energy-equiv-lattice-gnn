from argparse import Namespace

import torch
import torch_geometric.nn as pyg_nn

from gnn.mace import get_edge_vectors_and_lengths

class NNConvNet(torch.nn.Module):
    def __init__(self, params: Namespace) -> "NNConvNet":
        super().__init__()
        self.register_buffer('indices', torch.tensor([
            [0,  1,  2,  3,  4,  5],
            [1,  6,  7,  8,  9, 10],
            [2,  7, 11, 12, 13, 14],
            [3,  8, 12, 15, 16, 17],
            [4,  9, 13, 16, 18, 19],
            [5, 10, 14, 17, 19, 20]
        ])
        )
        
        
        self.dim_inner = params.hidden_irreps
        edge_dim_inner = params.hidden_irreps
        num_edge_ft = 5
        num_graph_ft = 21
        self.num_mp = params.message_passes

        self.node_ft_embedding = torch.nn.Linear(1, self.dim_inner)        
        
        self.conv_list = torch.nn.ModuleList()
        for i in range(self.num_mp):
            self.conv_list.append(pyg_nn.NNConv(self.dim_inner, self.dim_inner,
                                                nn=torch.nn.Sequential(
                                                    pyg_nn.Linear(num_edge_ft, edge_dim_inner),
                                                    torch.nn.ReLU(),
                                                    pyg_nn.Linear(edge_dim_inner, edge_dim_inner),
                                                    torch.nn.ReLU(),
                                                    pyg_nn.Linear(edge_dim_inner, 
                                                                self.dim_inner*self.dim_inner),
                                                    torch.nn.ReLU()
                                                ),
                                                aggr='add', 
                                                root_weight=False
                                                )
                    )

        self.pooling_fun = pyg_nn.global_mean_pool

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(self.dim_inner, 128),
            torch.nn.SELU(),
            torch.nn.Linear(128, 64),
            torch.nn.SELU(),
            torch.nn.Linear(64, 32),
            torch.nn.SELU(),
            torch.nn.Linear(32, num_graph_ft)
        )

        self.params = params

    def forward(self, batch):
        edge_index = batch.edge_index
        node_ft = batch.node_attrs

        node_ft = self.node_ft_embedding(node_ft)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch.positions, edge_index=edge_index, shifts=batch.shifts
        )
        edge_radii = batch.edge_attr
        edge_ft = torch.cat((vectors, lengths, edge_radii), dim=1)

        for i in range(self.num_mp):
            m = self.conv_list[i](node_ft, batch.edge_index, edge_ft)
            m = torch.nn.functional.relu(m)
            node_ft = node_ft + m      
        
        out = self.pooling_fun(node_ft, batch.batch)

        out = self.seq(out)
        C = out[:, self.indices]

        if self.params.positive=='square':
            C_pos = torch.linalg.matrix_power(C, 2)
        elif self.params.positive=='none':
            C_pos = C
        
        return {'stiffness': C_pos}
        