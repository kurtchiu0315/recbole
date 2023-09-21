# @Time   : 2021/11/29
# @Author : Ray Wu
# @Email  : ray7102ray7102@gmail.com

import argparse
import os
from recbole.utils.utils import init_wandb
import wandb
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import random

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    # wandb initialization
    init_wandb(config['wandb_project'], config['wandb_entity'], config)

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)

    
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

   
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)


    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    for k, v in test_result.items():
        wandb.run.summary[f'test/{k}'] = v

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    torch.set_num_threads(6)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='', help='name of datasets')
    parser.add_argument('--train_stage', type=str, default='', help='training stage')
    parser.add_argument('--pre_model_path', type=str, default='', help='pre_model for fine-tuning')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files \
                            else [os.path.join('config', 'model', f'{args.model}.yaml')]
    config_file_list.append(os.path.join('config', 'Default.yaml'))
    if os.path.exists(os.path.join('config', 'dataset', f'{args.dataset}.yaml')):
        config_file_list.append(os.path.join('config', 'dataset', f'{args.dataset}.yaml'))

    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
