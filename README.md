# AUtofocused oracles for model-based optimization

This repository is the official implementation of [Autofocused oracles for model-based optimization](https://arxiv.org/abs/2006.08052).

C. Fannjiang, J. Listgarten. AUtofocused oracles for model-based optimization. NeurIPS 2020.

## Requirements

## Training and Evaluation

Notebooks for running the superconductor design experiments (e.g., Table 1 in the paper) are superconductivity_groundtruth.ipynb and
superconductivity.ipynb. For the 1-D illustrative example, see toy.ipynb.

## Pre-trained Models

For the superconductor design experiments, pre-trained initial oracles can be found in initial_oracles. The ground-truth model is
gt_dim60.model, and the initial search model is saved in init_searchmodel.npz. See supeconductivity.ipynb for how to
reproduce and invoke these.

## Results

Superconductor design results are evaluated in results.ipynb, which reproduces Table 1 in the paper.
