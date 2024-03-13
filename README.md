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
    ├── benchmark_models.py: CGC, mCGC and NNConv models for benchmarking
    ├── run_train.py: train
    ├── eval_model.py: evaluate model checkpoint
```