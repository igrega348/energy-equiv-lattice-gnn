# %%
import sys
from pathlib import Path
folder = Path(__file__).parents[1]
if str(folder) not in sys.path:
    sys.path.insert(0, str(folder))
from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelSummary, 
    ModelCheckpoint, 
    EarlyStopping
) 
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch_geometric.loader import DataLoader

from gnn import EnergyEquivGNN
from train_utils import load_datasets, obtain_errors, aggr_errors, LightningWrappedModel
# %%
def main():
    params = Namespace(
        # network
        lmax=4,
        hidden_irreps='32x0e+32x1o+32x2e+32x3o+32x4e',
        readout_irreps='16x0e+16x1o+16x2e+16x3o+16x4e',
        num_edge_bases=6,
        interaction_reduction='sum',
        interaction_bias=True,
        agg_norm_const=4.0,
        inter_MLP_dim=64,
        inter_MLP_layers=3,
        correlation=3,
        global_reduction='mean',
        message_passes=2,
        positive_function='matrix_power_2',
        # training
        rotate=False,
        batch_size=64,
        valid_batch_size=64,
        log_every_n_steps=25,
        optimizer='adamw',
        lr=0.001, 
        amsgrad=True,
        weight_decay=1e-8,
        beta1=0.9,
        epsilon=1e-8,
        num_workers=4,
    )

    run_name = 'mace+ve'
    log_dir = Path(f'./{run_name}')
    log_dir.mkdir(exist_ok=True)
    rank_zero_info(log_dir)
    params.log_dir = str(log_dir)

    ############# setup data ##############
    train_dset = load_datasets(which='0imp', tag='train', reldens_norm=True)
    valid_dset = load_datasets(which='0imp', tag='valid', reldens_norm=True)

    max_edge_radius = train_dset.data.edge_attr.max().item()
    params.max_edge_radius = max_edge_radius
    # randomize the order of the dataset into loader
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
    lightning_model = LightningWrappedModel(EnergyEquivGNN, params)

    ############# setup trainer ##############
    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.3f}', every_n_epochs=1, monitor='val_loss', save_top_k=1),
        EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='min', strict=False) 
    ]
    trainer = pl.Trainer(
        accelerator='auto',
        accumulate_grad_batches=4, # effective batch size 256
        gradient_clip_val=10.0,
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
    trainer.fit(lightning_model, train_loader, valid_loader)

    ############# run testing ##############
    rank_zero_info('Testing')
    test_dset = load_datasets(tag='test', reldens_norm=False)
    test_loader = DataLoader(
        dataset=test_dset, batch_size=params.valid_batch_size, 
        shuffle=False, 
    )
    test_results = trainer.predict(lightning_model, test_loader, return_predictions=True, ckpt_path='best')
    df_errors = obtain_errors(test_results, 'test')
    eval_params = aggr_errors(df_errors)
    pd.Series(eval_params, name=run_name).to_csv(log_dir/f'aggr_results-step={trainer.global_step}.csv')

if __name__=='__main__':
    main()
