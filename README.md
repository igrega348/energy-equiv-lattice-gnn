# energy-equiv-lattice-gnn
Energy-conserving equivariant GNN for elasticity of lattice architected metamaterials

Work presented at the International Conference on Learning Representations (ICLR) 2024.

Link to paper:

https://openreview.net/forum?id=smy4DsUbBo

https://arxiv.org/abs/2401.16914

## Code structure
```
├── lattices: submodule for lattice processing, elasticity and plotting functions
├── gnn: ML modules
    ├── ...
├── scripts
    ├── benchmark_models: CGC, mCGC and NNConv models for benchmarking
    ├── train_utils.py: utilities for training
    ├── train_main.py: training script for the main model - EnergyEquivGNN (Energy-conserving equivariant GNN)
    ├── train_cgc_vanilla.py: train base CGC model for benchmarking
    ├── train_cgc_modified.py: train improved CGC model for benchmarking
    ├── train_nnconv.py: train NNConv based model for benchmarking
```

## Usage
Set up environment using requirement.txt or environment.yml file.

Data is available for download from https://doi.org/10.17863/CAM.106854

Try the scripts in `scripts` folder.
