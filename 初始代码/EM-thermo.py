!pip install chemprop -q
!pip install rdkit-pypi -q  # should be included in above after Chemprop v1.6 release

# Download test files from GitHub
!apt install subversion

import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
CUDA_VISIBLE_DEVICES=0
!nvidia-smi

#Train
arguments = [
    '--data_path', 'XX.csv',
    '--dataset_type', 'classification',
    '--metric', 'accuracy',
    '--extra_metrics', 'auc','f1','binary_cross_entropy','prc-auc','mcc',
    '--num_folds','5',
    '--features_generator','morgan',
    # '--features_path', 'XX.csv',
    '--save_dir', 'XX/',
    '--epochs', '500',
    '--save_smiles_splits'
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)


#Transfer learning
arguments = [
    '--data_path', 'XX.csv',
    '--dataset_type', 'classification',
    '--metric', 'accuracy',
    '--extra_metrics', 'auc','f1','binary_cross_entropy','prc-auc','mcc',
    '--num_folds','1',
    '--save_dir', 'XX',
    '--epochs', '500',
    '--checkpoint_frzn', 'XX/model.pt'
    # '--save_smiles_splits'
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)