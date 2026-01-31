"""Configuration namedtuples for experiment pipeline."""
from collections import namedtuple

# Raw data split from dataset loader
DataSplit = namedtuple('DataSplit', [
    'X_tr', 'y_tr',
    'X_val', 'y_val',
    'X_te', 'y_te'
])

# Output from projection evaluation (passed to NN evaluation)
ProjectionContext = namedtuple('ProjectionContext', [
    'reducer',
    'Z_tr', 'Z_val', 'Z_te',
    'supports_transform',
    'X_base', 'Z_base', 'X_noisy_per_class'
])

# Experiment configuration
ExperimentConfig = namedtuple('ExperimentConfig', [
    'sigma', 'n_samples', 'sigmas_cq', 'n_samples_cq'
])

# Configuration types
DatasetConfig = namedtuple('DatasetConfig', ['name', 'load_fn', 'input_dim', 'clip_bounds'])
ProjectionConfig = namedtuple('ProjectionConfig', ['name', 'setup'])
ModelConfig = namedtuple('ModelConfig', ['use_spec', 'hidden_dim', 'n_hidden', 'use_jac', 'lambda_jac'])
