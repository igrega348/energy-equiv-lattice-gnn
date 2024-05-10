# %%
import os
import sys
from pathlib import Path
folder = Path(__file__).parents[1]
if str(folder) not in sys.path:
    sys.path.insert(0, str(folder))
from argparse import Namespace
import json
from typing import Any, Optional, Tuple, Callable, Union
import time

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    EarlyStopping
) 
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from e3nn import o3

from benchmark_models.cgc_vanilla import CrystGraphConv
from gnn.datasets import GLAMM_Dataset as GLAMM_DSet
from lattices.lattices import elasticity_func
from train_utils import obtain_errors, aggr_errors, RotateLat
# %%
class GLAMM_Dataset(GLAMM_DSet):
    def __init__(self, 
            root: str, 
            catalogue_path: str,
            dset_fname: str,
            representation: str = 'fund_inner',
            node_ft: str = 'ones',
            edge_ft: str = 'r',
            graph_ft_format: str = 'cartesian_4',
            n_reldens: int = 1,
            choose_reldens: str = 'first',
            multiprocessing: Optional[Union[bool, int]] = False,
            regex_filter: Optional[str] = None,
            #
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
        ):
        super().__init__(
            root, catalogue_path, dset_fname, 
            representation, node_ft, edge_ft, graph_ft_format, 
            n_reldens, choose_reldens, multiprocessing, regex_filter, 
            transform, pre_transform, pre_filter
        )

    @staticmethod
    def mirror_cart_4(x: Tensor, axis: int):
        x_out = x.clone() # [num_graphs, 3, 3, 3, 3]
        assert axis in [0,1,2]
        if axis == 0:
            a = 4; b = 5
        elif axis == 1:
            a = 3; b = 4
        elif axis == 2:
            a = 3; b = 5

        for i2 in range(6):
            if i2<3:
                i = j = i2
            elif i2 == 3:
                i = 1; j = 2
            elif i2 == 4:
                i = 0; j = 2
            elif i2 == 5:
                i = 0; j = 1
            for j2 in range(i2,6):
                if j2<3:
                    k = l = j2
                elif j2 == 3:
                    k = 1; l = 2
                elif j2 == 4:
                    k = 0; l = 2
                elif j2 == 5:
                    k = 0; l = 1
                
                if (i == a) ^ (j == b):
                    val = x[...,i,j,k,l]
                    x_out[...,i,j,k,l] = -val
                    x_out[...,j,i,k,l] = -val
                    x_out[...,i,j,l,k] = -val
                    x_out[...,j,i,l,k] = -val
                    x_out[...,k,l,i,j] = -val
                    x_out[...,l,k,i,j] = -val
                    x_out[...,k,l,j,i] = -val
                    x_out[...,l,k,j,i] = -val
        return x_out

    @staticmethod
    def mirror_vector(x: Tensor, axis: int):
        x_out = x.clone()
        assert axis in [0,1,2]
        x_out[...,axis] = -x_out[...,axis]
        return x_out

    @staticmethod
    def process_one(lat_data, edge_ft_format, graph_ft_format, reldens_slice, pre_filter, pre_transform):
        out_list = GLAMM_DSet.process_one(lat_data, edge_ft_format, graph_ft_format, reldens_slice, pre_filter, pre_transform)
        # now apply augmentation to each
        new_out_list = []
        for lat in out_list:
            # copy original
            new_out_list.append(lat)
            # 3 rotations
            angle = torch.tensor(torch.pi/2)
            Qs = [o3.matrix_x(angle), o3.matrix_y(angle), o3.matrix_z(angle)]
            new_out_list.extend([
                Data(
                    node_attrs=lat.node_attrs,
                    edge_attr=lat.edge_attr,
                    edge_index=lat.edge_index,
                    positions = torch.einsum('ij,pj->pi', Q, lat.positions),
                    shifts = torch.einsum('ij,pj->pi', Q, lat.shifts),
                    unit_shifts = lat.unit_shifts,
                    rel_dens=lat.rel_dens,
                    stiffness=torch.einsum('...ijkl,ai,bj,ck,dl->...abcd', lat.stiffness.float(), Q, Q, Q, Q),
                    compliance=torch.einsum('...ijkl,ai,bj,ck,dl->...abcd', lat.compliance.float(), Q, Q, Q, Q),
                    name = lat.name
                ) for Q in Qs]
            )
            # 3 mirrors
            new_out_list.extend([
                Data(
                    node_attrs=lat.node_attrs,
                    edge_attr=lat.edge_attr,
                    edge_index=lat.edge_index,
                    positions = GLAMM_Dataset.mirror_vector(lat.positions, axis),
                    shifts = GLAMM_Dataset.mirror_vector(lat.shifts, axis),
                    unit_shifts = GLAMM_Dataset.mirror_vector(lat.unit_shifts, axis),
                    rel_dens=lat.rel_dens,
                    stiffness=GLAMM_Dataset.mirror_cart_4(lat.stiffness, axis),
                    compliance=GLAMM_Dataset.mirror_cart_4(lat.compliance, axis),
                    name = lat.name
                ) for axis in [0,1,2]]
            )
        return new_out_list
# %%
class LightningWrappedModel(pl.LightningModule):
    _time_metrics = {}
    inds_val = [[ 0,  1,  2,  3,  4,  5],
                [ 1,  6,  7,  8,  9, 10],
                [ 2,  7, 11, 12, 13, 14],
                [ 3,  8, 12, 15, 16, 17],
                [ 4,  9, 13, 16, 18, 19],
                [ 5, 10, 14, 17, 19, 20]]
    
    def __init__(self, model: torch.nn.Module, params: Namespace, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(params, dict):
            params = Namespace(**params)
        self.params = params
        self.model = model(params)

        self.register_buffer('stiffness_min', torch.zeros(1,21))
        self.register_buffer('stiffness_max', torch.ones(1,21))
       
        self.save_hyperparameters(params)

    def set_normalization(self, train_dset):
        stiffness = elasticity_func.stiffness_cart_4_to_Mandel(train_dset.data.stiffness)
        rows, cols = torch.triu_indices(6,6)
        target = stiffness[:, rows, cols] # [num_graphs, 21]
        # calculate min and max
        self.stiffness_min[:] = target.min(dim=0).values.view(1,21) # [1,21]
        self.stiffness_max[:] = target.max(dim=0).values.view(1,21) # [1,21]

    def normalize_target(self, target):
        return (target - self.stiffness_min) / (self.stiffness_max - self.stiffness_min) # [num_graphs, 21]
        
    def unnormalize_prediction(self, prediction):
        return prediction * (self.stiffness_max - self.stiffness_min) + self.stiffness_min # [num_graphs, 21]

    def configure_optimizers(self):
        params = self.params
        assert params.optimizer == 'radam'
        optim = torch.optim.RAdam(params=self.model.parameters(), lr=params.lr, 
            betas=(params.beta1,0.999), eps=params.epsilon,
            weight_decay=params.weight_decay,)
        return optim

    def training_step(self, batch, batch_idx):
        output = self.model(batch)

        rows, cols = torch.triu_indices(6,6)

        true_stiffness = batch['stiffness'] # [batch_size, 6, 6]
        pred_stiffness = output['stiffness'] # [batch_size, 21]

        target = self.normalize_target(true_stiffness[:, rows, cols])
        predicted = pred_stiffness

        stiffness_loss = torch.nn.functional.smooth_l1_loss(1000*predicted, 1000*target) 
        
        loss = stiffness_loss
    
        self.log('loss', loss, batch_size=batch.num_graphs, logger=True)
        self.log('stiffness_loss', stiffness_loss, batch_size=batch.num_graphs, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        directions = torch.randn(250, 3, dtype=torch.float32, device=batch.positions.device)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        
        output = self.model(batch)
        true_stiffness = batch['stiffness'] # [batch_size, 6, 6]
        pred_stiffness = output['stiffness'] # [batch_size, 21]
        pred_stiffness = self.unnormalize_prediction(pred_stiffness)[:, self.inds_val] # [batch_size, 6, 6]

        target = true_stiffness
        predicted = pred_stiffness
        stiffness_loss = torch.nn.functional.mse_loss(predicted, target)
        
        
        true_stiff_4 = elasticity_func.stiffness_Mandel_to_cart_4(true_stiffness)
        pred_stiff_4 = elasticity_func.stiffness_Mandel_to_cart_4(pred_stiffness)
        stiff_dir_true = torch.einsum('...ijkl,pi,pj,pk,pl->...p', true_stiff_4, directions, directions, directions, directions)       
        stiff_dir_pred = torch.einsum('...ijkl,pi,pj,pk,pl->...p', pred_stiff_4, directions, directions, directions, directions)
        stiff_dir_loss = torch.nn.functional.l1_loss(stiff_dir_pred, stiff_dir_true)
       
        loss = stiffness_loss
    
        self.log('val_loss', loss, batch_size=batch.num_graphs, logger=True, prog_bar=True, sync_dist=True)
        self.log('val_stiff_dir_loss', stiff_dir_loss, batch_size=batch.num_graphs, logger=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch: Any, batch_idx: int = 0, dataloader_idx: int = 0) -> Tuple:
        """Returns (prediction, true)"""
        prediction = self.unnormalize_prediction(self.model(batch)['stiffness'])[:, self.inds_val]
        return {'stiffness':prediction}, batch
    
    def on_train_epoch_start(self) -> None:
        self._time_metrics['_last_step'] = self.trainer.global_step
        self._time_metrics['_last_time'] = time.time()
        
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        step = self.trainer.global_step
        steps_done = step - self._time_metrics['_last_step']
        time_now = time.time()
        time_taken = time_now - self._time_metrics['_last_time']
        steps_per_sec = steps_done / time_taken
        self._time_metrics['_last_step'] = step
        self._time_metrics['_last_time'] = time_now
        self.log('steps_per_time', steps_per_sec, prog_bar=False, logger=True)
        # check if loss is nan
        loss = outputs['loss']
        if torch.isnan(loss):
            self.trainer.should_stop = True
            rank_zero_info('Loss is NaN. Stopping training')
# %%
def load_datasets(tag: str, which: str = '0imp', parent: str = '../../datasets', reldens_norm: bool = False, rotate: bool = False, augmented: bool = False):
    assert which in ['0imp_quarter', '0imp_half', '0imp', '1imp', '2imp', '4imp']
    if tag == 'test':
        root = os.path.join(parent, which)
        dset_file = 'test_cat.lat'
        processed_fname = 'test.pt'
    elif tag == 'train':
        root = os.path.join(parent, which)
        dset_file = 'training_cat.lat'
        if augmented:
            processed_fname = 'train_aug.pt'
        else:
            processed_fname = 'train.pt'
    elif tag == 'valid':
        root = os.path.join(parent, which)
        dset_file = 'validation_cat.lat'
        processed_fname = 'validation.pt'
    rank_zero_info(f'Loading dataset {tag} from {root}')
    dset = GLAMM_Dataset(
        root=root,
        catalogue_path=os.path.join(root, 'raw', dset_file),
        transform=RotateLat(rotate=rotate),
        dset_fname=processed_fname,
        n_reldens=3,
        choose_reldens='last',
        graph_ft_format='cartesian_4',
    )
    rank_zero_info(dset)

    # scaling and normalization
    if reldens_norm:
        normalization_factor = 10 / dset.data.rel_dens.view(-1,1,1,1,1)
    else:
        normalization_factor = 1000 # increased again because we're targeting relative densities on the order of 0.01

    dset.data.stiffness = (dset.data.stiffness * normalization_factor).float()
    dset.data.compliance = (dset.data.compliance / normalization_factor).float()

    return dset
# %%
def main():
    params = Namespace(
        # network
        hidden_irreps=64,
        interaction_reduction='sum',
        global_reduction='mean',
        message_passes=3,
        # training
        batch_size=256,
        valid_batch_size=512,
        log_every_n_steps=25,
        optimizer='radam',
        lr=0.001, 
        weight_decay=1e-8,
        beta1=0.9,
        epsilon=1e-8,
        num_workers=4,
    )

    run_name = 'cgc'
    log_dir = Path(f'./{run_name}')
    log_dir.mkdir(exist_ok=True)
    rank_zero_info(log_dir)
    params.log_dir = str(log_dir)

    ############# setup data ##############
    train_dset = load_datasets(which='0imp', augmented=False, tag='train', reldens_norm=False, rotate=False)
    valid_dset = load_datasets(which='0imp', tag='valid', reldens_norm=False, rotate=True)
    train_loader = DataLoader(
        dataset=train_dset, 
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dset,
        batch_size=params.valid_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
    )

    ############# setup model ##############
    lightning_model = LightningWrappedModel(CrystGraphConv, params)
    lightning_model.set_normalization(train_dset)

    ############# setup trainer ##############
    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.3f}', every_n_epochs=1, monitor='val_loss', save_top_k=1),
        EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='min', strict=False) 
    ]
    trainer = pl.Trainer(
        accelerator='auto',
        default_root_dir=params.log_dir,
        callbacks=callbacks,
        max_steps=50000,
        max_time='00:04:00:00',
        val_check_interval=100,
        log_every_n_steps=params.log_every_n_steps,
        check_val_every_n_epoch=None,
    )
    ############# save params ##############
    if trainer.is_global_zero:
        params_path = log_dir/f'params.json'
        params_path.write_text(json.dumps(vars(params), indent=2))

    ############# run training ##############
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    ############# run testing ##############
    rank_zero_info('Testing')
    test_dset = load_datasets(which='0imp', tag='test', reldens_norm=False)
    test_loader = DataLoader(test_dset, params.valid_batch_size, shuffle=False)
    test_results = trainer.predict(lightning_model, test_loader, return_predictions=True, ckpt_path='best')
    df_errors = obtain_errors(test_results, 'test')
    eval_params = aggr_errors(df_errors)
    pd.Series(eval_params, name=run_name).to_csv(log_dir/f'aggr_results-step={trainer.global_step}.csv')


if __name__=='__main__':
    main()
