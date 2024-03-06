from typing import Any, Optional, Tuple, Dict, Union
from argparse import Namespace
import time
import logging
from math import sqrt
import os
import shutil

import torch
from torch_scatter import scatter
from torch_geometric.data import Batch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from e3nn import o3
from e3nn.nn import BatchNorm, Dropout, FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
from e3nn.io import CartesianTensor

from .blocks import (
    RepeatNodeEmbedding,
    RadialEmbeddingBlock,
    FourierBasisEmbeddingBlock,
    GraphAttention,
    CGCLayer,
    MACELayer,
    GeneralLinearReadoutBlock,
    Cart_4_to_Mandel,
    VectorNormSelection,
    NodeConnectivityEmbedding,
    GeneralNonLinearReadoutBlock,
    TensorProductInteractionBlock,
    TensorProductResidualInteractionBlock,
    EquivariantProductBlock,
    OneTPReadoutBlock,
    GlobalSumHistoryPooling,
    GlobalElementwisePooling
)
from .mace import get_edge_vectors_and_lengths, reshape_irreps

class BaseModel(pl.LightningModule):
    _time_metrics: Dict
    run_stats: Dict

    def __init__(self, params:Namespace, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params = params
        self._time_metrics = {}
        self.run_stats = {'loss':[]}


    def configure_optimizers(self):
        params = self.params

        if (params.optimizer).lower()=='adamw':
            rank_zero_info('Setting optimizer AdamW')
            optimizer = torch.optim.AdamW(
                params=self.parameters(), lr=params.lr, 
                betas=(params.beta1,0.999), eps=params.epsilon,
                amsgrad=params.amsgrad, weight_decay=params.weight_decay,
            )
        elif (params.optimizer).lower()=='nadam':
            rank_zero_info('Setting optimizer NAdam')
            optimizer = torch.optim.NAdam(
                params=self.parameters(), lr=params.lr, 
                weight_decay=params.weight_decay,
            )
        elif (params.optimizer).lower()=='radam':
            rank_zero_info('Setting optimizer RAdam')
            optimizer = torch.optim.RAdam(
                params=self.parameters(), lr=params.lr, betas=(params.beta1,0.999),
                weight_decay=params.weight_decay,
            )            
        elif (params.optimizer).lower()=='sgd':
            rank_zero_info('Setting optimizer SGD')
            optimizer = torch.optim.SGD(
                params=self.parameters(), lr=params.lr, 
                momentum=params.momentum, nesterov=params.nesterov,
                weight_decay=params.weight_decay,
            )

        if not hasattr(params, 'scheduler') or not isinstance(params.scheduler, str):
            lr_scheduler = None
        elif (params.scheduler).lower()=="linearlr":
            rank_zero_info('Setting scheduler LinearLR')
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer, start_factor=1, end_factor=0.1,
                total_iters=params.max_num_epochs
            )
        elif (params.scheduler).lower()=="steplr":
            rank_zero_info('Setting scheduler StepLR')
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, params.lr_step_size, gamma=0.5)
        elif (params.scheduler).lower()=="multisteplr":
            rank_zero_info('Setting scheduler MultiStepLR')
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.lr_milestones, gamma=0.2)

        if lr_scheduler is not None:
            return {"optimizer": optimizer, 'lr_scheduler':{
                'scheduler': lr_scheduler,
                'interval':'epoch', 'frequency':1
            }}  
        else:
            return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = torch.nn.functional.mse_loss(
            self.loss_weights*output['stiffness'], self.loss_weights*batch['stiffness']
        )

        # calculate 'percentage' error for each row of the output
        vals, _ = batch['stiffness'].abs().max(dim=1, keepdim=True)
        error = torch.mean((output['stiffness']-batch['stiffness']).abs()/vals, dim=1)
        train_err = torch.mean(error)

        self.log("loss", loss, 
            prog_bar=False, batch_size=batch.num_graphs,
            # on_step=False, on_epoch=True
            )
        self.log('train_err', train_err, 
            prog_bar=False, batch_size=batch.num_graphs, sync_dist=True
        )
        self.log("lr", self.optimizers().param_groups[0]['lr'], 
                prog_bar=False, batch_size=batch.num_graphs, 
                on_epoch=True, on_step=False, sync_dist=True
                )

        return loss    

    def validation_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self(batch)

        loss = torch.nn.functional.mse_loss(
            self.loss_weights*output['stiffness'], self.loss_weights*batch['stiffness']
        )
        # calculate 'percentage' error for each row of the output
        vals, _ = batch['stiffness'].abs().max(dim=1, keepdim=True)
        error = torch.mean((output['stiffness']-batch['stiffness']).abs()/vals, dim=1)
        val_err = torch.mean(error)

        self.log("val_loss", loss, 
            prog_bar=True, batch_size=batch.num_graphs, sync_dist=True
        )
        self.log('val_err', val_err, 
            prog_bar=True, batch_size=batch.num_graphs, sync_dist=True
        )
        return output['stiffness'], batch['stiffness']

    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Tuple:
        """Returns (prediction, true)"""
        return self(batch), batch

    def on_train_start(self) -> None:
        self._time_metrics['_train_start_time'] = time.time()
        self._time_metrics['_train_start_step'] = self.global_step

    def on_train_epoch_start(self) -> None:
        self._time_metrics['_train_epoch_start_time'] = time.time()
        self._time_metrics['_train_epoch_start_step'] = self.global_step

    def on_train_epoch_end(self) -> None:
        tn = time.time()
        epoch_now = self.current_epoch + 1
        step_now = self.global_step
        time_per_epoch = (tn - self._time_metrics['_train_start_time'])/epoch_now
        epoch_steps = step_now - self._time_metrics['_train_epoch_start_step']
        total_steps = step_now - self._time_metrics['_train_start_step']
        time_per_step_total = (tn - self._time_metrics['_train_start_time'])/total_steps
        time_per_step_epoch = (tn - self._time_metrics['_train_epoch_start_time'])/epoch_steps
        
        if self.trainer.max_epochs>0:
            max_epochs = self.trainer.max_epochs
            eta = (max_epochs-epoch_now)*time_per_epoch
        elif self.trainer.max_steps>0:
            max_steps = self.trainer.max_steps
            eta = (max_steps - step_now)*time_per_step_total

        self.log("eta", eta,
                prog_bar=True, sync_dist=True
                )
        self.log('step_per_time', 1/time_per_step_epoch,
                prog_bar=False, sync_dist=True
                )


class LatticeGNN(BaseModel):

    def __init__(
        self,
        params: Namespace, 
        *args: Any, 
        **kwargs: Any
    ) -> "LatticeGNN":
        super().__init__(params, *args, **kwargs)
        self.params = params

        hidden_irreps = o3.Irreps(params.hidden_irreps)

        self.register_buffer('loss_weights', 10*torch.ones((1,21)))     
        
        self.node_ft_embedding = torch.nn.Linear(in_features=1, out_features=hidden_irreps.count(o3.Irrep(0,1)))
        self.number_of_edge_basis = 6
        node_ft_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.number_of_edge_basis*2}x0e")
        edge_attr_irreps = o3.Irreps.spherical_harmonics(params.lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            edge_attr_irreps,
            normalize=True, normalization='component'
        )
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (edge_attr_irreps * num_features).sort()[0].simplify()
        readout_irreps = o3.Irreps(params.readout_irreps)

        self.linear_skip = o3.Linear(
                irreps_in=hidden_irreps,
                irreps_out=hidden_irreps,
                internal_weights=True,
                shared_weights=True
            )

        self.gnn_layers = torch.nn.ModuleList([
            MACELayer(node_ft_irreps, edge_attr_irreps, edge_feats_irreps, interaction_irreps, hidden_irreps, params.agg_norm_const, params.interaction_reduction, True, params.correlation),
            MACELayer(hidden_irreps, edge_attr_irreps, edge_feats_irreps, interaction_irreps, hidden_irreps, params.agg_norm_const, params.interaction_reduction, True, params.correlation),
        ])

        self.readout = GeneralNonLinearReadoutBlock(
                irreps_in=hidden_irreps,
                irreps_out=readout_irreps,
                hidden_irreps=readout_irreps,
                gate=torch.nn.functional.silu,
            )       
        
        self.pooling = GlobalSumHistoryPooling(reduce=params.global_reduction)
        self.linear = o3.Linear(readout_irreps, readout_irreps, 
            internal_weights=True, 
            shared_weights=True,
            biases=True
        )
        self.fourth_order_expansion = OneTPReadoutBlock(
            irreps_in=readout_irreps,
            irreps_out=o3.Irreps('2x0e+2x2e+1x4e')
        )

        self.save_hyperparameters()

    def forward(
        self,
        batch: Batch
    ) -> Dict:
        
        num_graphs = batch.num_graphs
        edge_index = batch.edge_index
        node_ft = batch.node_attrs
        node_ft = self.node_ft_embedding(node_ft)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch.positions, edge_index=edge_index, shifts=batch.shifts
        )
        edge_length_embedding = soft_one_hot_linspace(
            lengths.squeeze(-1), start=0, end=0.6, number=self.number_of_edge_basis, basis='fourier', cutoff=False
        )
        edge_radii = batch.edge_attr
        edge_radius_embedding = soft_one_hot_linspace(
            edge_radii.squeeze(-1), 0, 0.03, self.number_of_edge_basis, 'fourier', False
        )
        edge_feats = torch.cat(
            (edge_length_embedding, edge_radius_embedding), 
            dim=1
        )
        edge_sh = self.spherical_harmonics(vectors)
        
        outputs = []
        node_ft = self.gnn_layers[0](node_ft, edge_index, edge_sh, edge_feats)

        for i_mp in range(self.params.message_passes-1):
            node_ft = self.gnn_layers[1](node_ft, edge_index, edge_sh, edge_feats) + self.linear_skip(node_ft)
        
        outputs.append(self.readout(node_ft))
       
        outputs = torch.stack(outputs, dim=-1)

        graph_ft = self.pooling(outputs, batch.batch, num_graphs)
        graph_ft = self.linear(graph_ft)
        stiffness = self.fourth_order_expansion(graph_ft) 

        # if torch.sum(torch.isnan(stiffness))>0:
        #     raise ValueError('nan in stiffness')

        return {'stiffness': stiffness}


class PositiveGNN(LatticeGNN):
    def __init__(self, params: Namespace, *args: Any, **kwargs: Any) -> "PositiveGNN":
        super().__init__(params, *args, **kwargs)
        self.el_tens = CartesianTensor('ijkl=ijlk=jikl=klij')
        self.cart_to_Mandel = Cart_4_to_Mandel()
        if params.func.lower()=='exp':
            self.matrix_func = torch.linalg.matrix_exp
        elif params.func.lower()=='square':
            self.matrix_func = lambda x: torch.linalg.matrix_power(x, 2)
        
        self.run_stats['stiffness_loss'] = []
        # self.run_stats['compliance_loss'] = []
        # self.run_stats['stiffness_rel_loss'] = []
        # self.run_stats['eig_loss'] = []
        # self.run_stats['eig_loss_inv'] = []
        self._prev_checkpoint_step = None
        self._prev_checkpoint_fn = None


    def forward(self, batch: Batch) -> Dict:
        out = super().forward(batch)['stiffness']

        stiffness = self.el_tens.to_cartesian(out)
        C = self.cart_to_Mandel(stiffness)
        C_exp = self.matrix_func(C)

        # if torch.sum(torch.isnan(C_exp))>0:
        #     raise ValueError('nan in C_predictions')

        return {'stiffness': C_exp}

    def training_step(self, batch, batch_idx):
        output = self(batch)

        if torch.sum(torch.isnan(output['stiffness']))>0:
            rank_zero_info('nan in output. Loading old weights')
            self.load_state_dict(torch.load(self._prev_checkpoint_fn)['state_dict'])
            output = self(batch)

        rows, cols = torch.triu_indices(6,6)
        loss = torch.nn.functional.mse_loss(
            output['stiffness'][:, rows, cols], batch['stiffness'][:, rows, cols]
        )
        self.run_stats['stiffness_loss'].append((self.global_step, loss.detach().item()))
     
        self.log("loss", loss, 
            prog_bar=False, batch_size=batch.num_graphs,
            # on_step=False, on_epoch=True
            )
       
        self.log("lr", self.optimizers().param_groups[0]['lr'], 
                prog_bar=False, batch_size=batch.num_graphs, 
                on_epoch=True, on_step=False, sync_dist=True
                )

        self.run_stats['loss'].append((self.global_step, loss.detach().item()))

        return loss   

    
    def validation_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self(batch)

        loss = torch.nn.functional.mse_loss(
            output['stiffness'], batch['stiffness']
        )
      
        self.log("val_loss", loss, 
            prog_bar=True, batch_size=batch.num_graphs, sync_dist=True
        )
        return output['stiffness'], batch['stiffness']

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        last_fn = self.trainer.checkpoint_callback.last_model_path
        if (len(last_fn)>0) and (os.path.exists(last_fn)) and (self._prev_checkpoint_step != self.global_step) and (torch.isnan(self.state_dict()['node_ft_embedding.weight']).sum().item()==0):
            new_fn = last_fn.replace('last.ckpt', 'previous.ckpt')
            shutil.copyfile(last_fn, new_fn)
            self._prev_checkpoint_step = self.global_step
            self._prev_checkpoint_fn = new_fn


class SpectralGNN(BaseModel):
    inds_val = [[ 0,  1,  2,  3,  4,  5],
                [ 1,  6,  7,  8,  9, 10],
                [ 2,  7, 11, 12, 13, 14],
                [ 3,  8, 12, 15, 16, 17],
                [ 4,  9, 13, 16, 18, 19],
                [ 5, 10, 14, 17, 19, 20]]
    inds_Q = [[0,3,4],[3,1,5],[4,5,2]]

    def __init__(
        self,
        params: Namespace, 
        *args: Any, 
        **kwargs: Any
    ) -> "SpectralGNN":
        super().__init__(params, *args, **kwargs)
        self.params = params

        hidden_irreps = o3.Irreps(params.hidden_irreps)

        self.node_ft_embedding = torch.nn.Linear(in_features=1, out_features=hidden_irreps.count(o3.Irrep(0,1)))
        self.number_of_edge_basis = 6
        node_ft_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.number_of_edge_basis*2}x0e")
        edge_attr_irreps = o3.Irreps.spherical_harmonics(params.lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            edge_attr_irreps,
            normalize=True, normalization='component'
        )
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (edge_attr_irreps * num_features).sort()[0].simplify()
        readout_irreps = o3.Irreps(params.readout_irreps)

        self.linear_skip = o3.Linear(
                irreps_in=hidden_irreps,
                irreps_out=hidden_irreps,
                internal_weights=True,
                shared_weights=True
            )

        self.gnn_layers = torch.nn.ModuleList([
            MACELayer(node_ft_irreps, edge_attr_irreps, edge_feats_irreps, interaction_irreps, hidden_irreps, params.agg_norm_const, params.interaction_reduction, True, params.correlation),
            MACELayer(hidden_irreps, edge_attr_irreps, edge_feats_irreps, interaction_irreps, hidden_irreps, params.agg_norm_const, params.interaction_reduction, True, params.correlation),
        ])

        self.lin_readout = GeneralLinearReadoutBlock(
            irreps_in=hidden_irreps,
            hidden_irreps=hidden_irreps,
            irreps_out=readout_irreps,
        )

        self.nonlin_readout = GeneralNonLinearReadoutBlock(
            irreps_in=hidden_irreps,
            irreps_out=readout_irreps,
            hidden_irreps=readout_irreps,
            gate=torch.nn.functional.silu,
        )       
        
        self.pooling = GlobalSumHistoryPooling(reduce=params.global_reduction)
        self.linear = o3.Linear(readout_irreps, readout_irreps, 
            internal_weights=True, 
            shared_weights=True,
            biases=True
        )

        # self.vec_select = VectorNormSelection(num_vecs_in=8, num_vecs_out=2)
        self.sym_tensor = CartesianTensor('ij=ji')
        self.stack_irreps = reshape_irreps(readout_irreps)

        s = 1/sqrt(2.0)
        self.register_buffer('tens_multipliers', torch.tensor([[1,s,s],[s,1,s],[s,s,1]], dtype=torch.float32))
        self.save_hyperparameters()
        
    def forward(
        self,
        batch: Batch
    ) -> Dict:
        
        num_graphs = batch.num_graphs
        edge_index = batch.edge_index
        node_ft = batch.node_attrs
        node_ft = self.node_ft_embedding(node_ft)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch.positions, edge_index=edge_index, shifts=batch.shifts
        )
        edge_length_embedding = soft_one_hot_linspace(
            lengths.squeeze(-1), start=0, end=0.6, number=self.number_of_edge_basis, basis='fourier', cutoff=False
        )
        edge_radii = batch.edge_attr
        edge_radius_embedding = soft_one_hot_linspace(
            edge_radii.squeeze(-1), 0, 0.03, self.number_of_edge_basis, 'fourier', False
        )
        edge_feats = torch.cat(
            (edge_length_embedding, edge_radius_embedding), 
            dim=1
        )
        edge_sh = self.spherical_harmonics(vectors)
        
        outputs = []
        node_ft = self.gnn_layers[0](node_ft, edge_index, edge_sh, edge_feats)

        for i_mp in range(self.params.message_passes-1):
            node_ft = self.gnn_layers[1](node_ft, edge_index, edge_sh, edge_feats) + self.linear_skip(node_ft)
        
        outputs.append(self.lin_readout(node_ft))
        outputs.append(self.nonlin_readout(node_ft))
       
        outputs = torch.stack(outputs, dim=-1)

        graph_ft = self.pooling(outputs, batch.batch, num_graphs)
        graph_ft = self.linear(graph_ft)

        graph_ft = self.stack_irreps(graph_ft)
        graph_ft = graph_ft.sum(dim=1)
        graph_ft = self.sym_tensor.to_cartesian(graph_ft)
        # graph_ft[:, 21:] = graph_ft[:, 21:]

        # vals = graph_ft[:, :21]
        # vecs = graph_ft[:, 21:] 
        # vecs = self.vec_select(vecs) # [num_graphs, 2, 3]
        # e1 = vecs[:,0,:]
        # # # e1 = graph_ft[:, 21:24]
        # # # # e1 = e1 / (torch.linalg.norm(e1, dim=1, keepdim=True))
        # e2 = vecs[:,1,:]
        # # # e2 = graph_ft[:, 24:27]
        # # e2 = e2 / (torch.linalg.norm(e2, dim=1, keepdim=True))
        # e3 = torch.linalg.cross(e1, e2)
        # e2 = torch.linalg.cross(e3, e1)

        # R = torch.stack([e1,e2,e3], dim=1)
        # length = torch.linalg.norm(R, dim=2, keepdim=True)
        # R = R / length
        # # R = R / (length + 1e-8)
        # A = self.sym_tensor.to_cartesian(graph_ft[:,21:])
        # _, evecs = torch.linalg.eigh(A)   # columns are normalized eigenvectors
        # R = torch.permute(evecs, (0,2,1)) # rows of R are orthonormal vectors
        # # R = A
        # # Q, _ = torch.linalg.qr(A)
        # # R = torch.permute(Q, (0,2,1))

        # M = vals[:, self.inds_val]
        # _Q,_R = torch.linalg.qr(M)
        # _evals = _R[:,[0,1,2,3,4,5],[0,1,2,3,4,5]]


        # T = _Q[:,self.inds_Q,:]*self.tens_multipliers.view(1,3,3,1)
        # C_local = torch.einsum('pa,pa,pija,pkla->pijkl', _evals, _evals, T, T)
        # C_predictions = torch.einsum('pijkl,pia,pjb,pkc,pld->pabcd', C_local, R, R, R, R)
        # # C_predictions = C_local
        
        # if torch.sum(torch.isnan(C_predictions))>0:
        #     raise ValueError('nan in C_predictions')

        return {'stiffness': graph_ft}
        # return {'stiffness': C_predictions}

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = torch.nn.functional.mse_loss(
            10*output['stiffness'], 10*batch['stiffness']
        ) 

        self.log("loss", loss, 
            prog_bar=False, batch_size=batch.num_graphs,
            # on_step=False, on_epoch=True
            )
      
        self.log("lr", self.optimizers().param_groups[0]['lr'], 
                prog_bar=False, batch_size=batch.num_graphs, 
                on_epoch=True, on_step=False, sync_dist=True
                )
        return loss    

    def validation_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self(batch)

        loss = torch.nn.functional.mse_loss(
            output['stiffness'], batch['stiffness']
        )
      
        self.log("val_loss", loss, 
            prog_bar=True, batch_size=batch.num_graphs, sync_dist=True
        )
        return output['stiffness'], batch['stiffness']

class LatticeAttention(BaseModel):

    def __init__(
        self,
        params: Namespace, 
        *args: Any, 
        **kwargs: Any
    ) -> "LatticeAttention":
        super().__init__(params, *args, **kwargs)
        self.params = params

        hidden_irreps = o3.Irreps(params.hidden_irreps)

        self.register_buffer('loss_weights', torch.ones((1,21)))

        self.node_ft_embedding = torch.nn.Linear(in_features=1, out_features=hidden_irreps.count(o3.Irrep(0,1)))

        node_ft_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.number_of_edge_basis = 5
        number_of_basis = self.number_of_edge_basis*2
        edge_feats_irreps = o3.Irreps(f"{number_of_basis}x0e")
        irreps_sh = o3.Irreps.spherical_harmonics(params.lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            irreps_sh, normalize=True, normalization='component'
        )
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (irreps_sh * num_features).sort()[0].simplify()
        readout_irreps = o3.Irreps(params.readout_irreps)

        irreps_key = o3.Irreps('32x0e+32x1o')
        irreps_query = o3.Irreps('32x0e+32x1o')

        self.gat0 = GraphAttention(
            node_ft_irreps, irreps_query, irreps_key, hidden_irreps, irreps_sh, number_of_basis
        )
        
        self.gat1 = GraphAttention(
            hidden_irreps, irreps_query, irreps_key, hidden_irreps, irreps_sh, number_of_basis
        )

        self.pooling = GlobalSumHistoryPooling(reduce=params.global_reduction)
        self.linear = o3.Linear(hidden_irreps, readout_irreps, 
            internal_weights=True, 
            shared_weights=True,
            biases=True
        )
        self.fourth_order_expansion = OneTPReadoutBlock(
            irreps_in=readout_irreps,
            irreps_out=o3.Irreps('2x0e+2x2e+1x4e')
        )

        self.save_hyperparameters()

    def forward(
        self,
        batch: Batch
    ) -> Dict:
        
        num_graphs = batch.num_graphs
        edge_index = batch.edge_index
        node_ft = batch.node_attrs
        # node_ft = self.connectivity_embedding(batch.node_attrs, edge_index)
        node_ft = self.node_ft_embedding(node_ft)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch.positions, edge_index=edge_index, shifts=batch.shifts
        )
        lengths = lengths.squeeze(-1)
        edge_length_embedding = soft_one_hot_linspace(
            lengths, start=0, end=0.6, number=self.number_of_edge_basis, basis='fourier', cutoff=False
        )
        edge_radius_embedding = soft_one_hot_linspace(
            batch.edge_attr.squeeze(-1), 0, 0.02, self.number_of_edge_basis, 'fourier', False
        )
        edge_feats = torch.cat(
            (edge_length_embedding, edge_radius_embedding), 
            dim=1
        )
        edge_sh = self.spherical_harmonics(vectors)
        
        outputs = []

        node_ft = self.gat0(node_ft, edge_index, edge_sh, edge_feats)
        for _ in range(self.params.message_passes - 1):
            node_ft = self.gat1(node_ft, edge_index, edge_sh, edge_feats)

        outputs.append(node_ft)

        outputs = torch.stack(outputs, dim=-1)

        graph_ft = self.pooling(outputs, batch.batch, num_graphs)
        # TODO: add dropout
        graph_ft = self.linear(graph_ft)
        stiffness = self.fourth_order_expansion(graph_ft) # [num_]

        return {'stiffness': stiffness}

class CrystGraphConv(BaseModel):
    def __init__(
        self,
        params: Namespace, 
        *args: Any, 
        **kwargs: Any
    ) -> "CrystGraphConv":
        super().__init__(params, *args, **kwargs)
        self.params = params

        self.register_buffer('loss_weights', 10*torch.ones((1,21)))
        
        hidden_dim = params.hidden_dim
        self.node_ft_embedding = torch.nn.Linear(in_features=3, out_features=hidden_dim)
        self.edge_ft_embedding = torch.nn.Linear(in_features=5, out_features=hidden_dim)
        
        self.cgc_layers = torch.nn.ModuleList(
            [CGCLayer(hidden_dim, hidden_dim, params.interaction_reduction) for i in range(params.message_passes)]
        )

        self.pooling = GlobalElementwisePooling(params.global_reduction)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 32),
            torch.nn.Softplus(),
            torch.nn.Linear(32, 21)
        )

        self.save_hyperparameters()

    def forward(
        self,
        batch: Batch
    ) -> Dict:
        
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
        
        graph_ft = self.pooling(node_ft, batch.batch, num_graphs) # [num_nodes, hidden_dim]

        stiffness = self.mlp(graph_ft) # [num_graphs, 21]

        return {'stiffness': stiffness}


class MCrystGraphConv(BaseModel):
    def __init__(
        self,
        params: Namespace, 
        *args: Any, 
        **kwargs: Any
    ) -> "MCrystGraphConv":
        super().__init__(params, *args, **kwargs)
        self.params = params

        self.register_buffer('loss_weights', 10*torch.ones((1,21)))
        
        hidden_dim = params.hidden_dim
        self.node_ft_embedding = torch.nn.Linear(in_features=3, out_features=hidden_dim)
        self.edge_ft_embedding = torch.nn.Linear(in_features=5, out_features=hidden_dim)
        self.line_edge_ft_embedding = torch.nn.Linear(in_features=5, out_features=hidden_dim)
        
        self.cgc_layers = torch.nn.ModuleList(
            [CGCLayer(hidden_dim, hidden_dim, params.interaction_reduction) for i in range(params.message_passes)]
        )
        self.m_cgc_layers = torch.nn.ModuleList(
            [CGCLayer(hidden_dim, hidden_dim, params.interaction_reduction) for i in range(params.message_passes)]
        )

        self.pooling = GlobalElementwisePooling(params.global_reduction)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.Softplus(),
            torch.nn.Linear(128, 64),
            torch.nn.Softplus(),
            torch.nn.Linear(64, 32),
            torch.nn.Softplus(),
            torch.nn.Linear(32, 21)
        )

        self.save_hyperparameters()

    def forward(
        self,
        batch: Batch
    ) -> Dict:
        
        num_graphs = batch.num_graphs
        edge_index = batch.edge_index
        line_edge_index = batch.line_index
        node_ft = batch.positions # [num_nodes, 3]
        edge_radius = batch.edge_attr # [num_edges, 1]

        node_ft = self.node_ft_embedding(node_ft) # [num_nodes, hidden_dim]

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=batch.positions, edge_index=edge_index, shifts=batch.shifts, normalize=True
        )
        edge_ft = torch.cat([vectors, lengths, edge_radius], dim=1) # [num_edges, 5]
        edge_ft = self.edge_ft_embedding(edge_ft) # [num_edges, hidden_dim]


        line_sender, line_receiver = line_edge_index
        line_edge_vec_sender = vectors[line_sender]
        line_edge_vec_receiver = vectors[line_receiver]
        line_edge_len_sender = lengths[line_sender]
        line_edge_len_receiver = lengths[line_receiver]
        line_edge_cross = torch.linalg.cross(line_edge_vec_sender, line_edge_vec_receiver, dim=1)
        line_edge_ft = torch.cat(
            [line_edge_cross/(line_edge_len_sender*line_edge_len_receiver), line_edge_len_sender, line_edge_len_receiver], 
            dim=1
        )

        line_edge_ft = self.line_edge_ft_embedding(line_edge_ft)

        for i_mp in range(self.params.message_passes):
            edge_ft = edge_ft + self.m_cgc_layers[i_mp](edge_ft, line_edge_index, line_edge_ft)
            node_ft = node_ft + self.cgc_layers[i_mp](node_ft, edge_index, edge_ft)
        
        graph_ft = self.pooling(node_ft, batch.batch, num_graphs) # [num_nodes, hidden_dim]

        stiffness = self.mlp(graph_ft) # [num_graphs, 21]

        return {'stiffness': stiffness}