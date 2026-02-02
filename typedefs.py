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

# Configuration types
DatasetConfig = namedtuple('DatasetConfig', ['name', 'load_fn', 'input_dim', 'clip_bounds', 'sigma', 'n_samples'])
ProjectionConfig = namedtuple('ProjectionConfig', ['name', 'setup'])
ModelConfig = namedtuple('ModelConfig', ['use_spec', 'hidden_dim', 'n_hidden', 'use_jac', 'lambda_jac'])

# Bundled train/val data for model training
TrainData = namedtuple('TrainData', ['X_tr', 'Z_tr', 'X_val', 'Z_val'])

# Output directory paths
OutputDirs = namedtuple('OutputDirs', ['models', 'images', 'results'])

# Context for a single experiment run (dataset + projection + seed)
RunContext = namedtuple('RunContext', ['dataset_cfg', 'projection_cfg', 'seed'])

# Training configuration
TrainingConfig = namedtuple('TrainingConfig', ['max_epochs', 'batch_size', 'lr', 'patience'])

# Training result
TrainingResult = namedtuple('TrainingResult', ['model', 'best_val_loss', 'final_train_loss', 'epochs', 'early_stopped', 'training_time'])
