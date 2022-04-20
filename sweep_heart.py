import wandb
import math
from fit_heart import fit

sweep_config = {'method': 'random'}
metric = {'name': 'val_acc', 'goal': 'maximize'}

# Tutorial of hyperparams sweep using wandb:
# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch
# /Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb

parameters_dict = {
    # Refer to https://docs.wandb.ai/guides/sweeps/configuration#distributions for distribution of hyperparams sweep
    'dataset': {
            # uniformly select between two datasets
            'distribution': 'categorical',
            'values': ['HEART']
        },
    'algorithm': {
                # uniformly select between two datasets
                'distribution': 'categorical',
                'values': ['SKLEARN']
            },
    'n_estimators': {
        # randomly select real number x in [min, max] and return exp(x)
        'distribution': 'int_uniform',
        'min': 30,
        'max': 80
    },
    'max_depth': {
        'distribution': 'int_uniform',
        'min': 4,
        'max': 10
    },
    'seed': {
        'distribution': 'constant',
        'value': 1
    },
    'verbose': {
            'distribution': 'constant',
            'value': True
        },
    'group_id': {
        'distribution': 'constant',
        'value': 3096
    },
    'sub_size': {
        'distribution': 'constant',
        'value': 5000
    },
    'bootstrap': {
        'distribution': 'constant',
        'value': True
    }
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict
if __name__ == '__main__':
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="sklearn-sweeps-demo")
    wandb.agent(sweep_id, fit, count=20)
