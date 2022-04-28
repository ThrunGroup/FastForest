import wandb
from heart_experiments.fit_heart import fit
from constants import SKLEARN, FASTFOREST,HEART

if __name__ == '__main__':
    # Tutorial of hyperparams sweep using wandb: shorturl.at/vwAFS
    sweep_config = {'method': 'random'}
    metric = {'name': 'val_acc', 'goal': 'maximize'}
    parameters_dict = {
        # Refer to https://docs.wandb.ai/guides/sweeps/configuration#distributions for distribution of hyperparams sweep
        'dataset': {
            # select heart_disease dataset
            'distribution': 'categorical',
            'values': [HEART]
        },
        'algorithm': {
            # select sklearn algorithm
            'distribution': 'categorical',
            'values': [SKLEARN]
        },
        'n_estimators': {
            # randomly select real number x in [min, max] and return int(x)
            'distribution': 'int_uniform',
            'min': 20,
            'max': 100
        },
        'max_depth': {
            # randomly select real number x in [min, max] and return int(x)
            'distribution': 'int_uniform',
            'min': 10,
            'max': 25
        },
        'seed': {
            'distribution': 'constant',
            'value': 1
        },
        'verbose': {
            'distribution': 'constant',
            'value': True
        },
        'sub_size': {
            'distribution': 'constant',
            'value': 3000
        },
        'bootstrap': {
            'distribution': 'constant',
            'value': True
        },
        'is_balanced': {
            'distribution': 'constant',
            'value': True
        }
    }

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="sklearn-sweeps-demo")
    wandb.agent(sweep_id, fit, count=20)
