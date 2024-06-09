# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29
# @Author : Zihan Lin, Yupeng Hou
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn

import sys
import os, torch
import argparse

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default=None, help='fixed config files')
    parser.add_argument('--params_file', '-p', type=str, default=None, help='parameters file')
    args, _ = parser.parse_known_args()
    
    try:
        model = [arg.split('=')[1] for arg in sys.argv if '--model=' in arg][0]
        dataset = [arg.split('=')[1] for arg in sys.argv if '--dataset=' in arg][0]
    except:
        raise NotImplementedError('Make sure model and dataset follow the format(--{k}={v}) in command line.')
    
    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    config_file_list = args.config_files.strip().split(' ') if args.config_files \
                            else [os.path.join('config', 'model', f'{model}.yaml')]
    config_file_list.append(os.path.join('config', 'Default.yaml'))
    if os.path.exists(os.path.join('config', 'dataset', f'{dataset}.yaml')):
        config_file_list.append(os.path.join('config', 'dataset', f'{dataset}.yaml'))

    params_file = args.params_file if args.params_file \
                        else os.path.join('config', 'hyper', f'{model}.hyper')
    export_result_file = os.path.join('saved', f'{model}-{dataset}.hyper.result')
    hp = HyperTuning(objective_function, algo='exhaustive',
                     params_file=params_file, fixed_config_file_list=config_file_list)
    hp.run()
    hp.export_result(export_result_file)
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    torch.set_num_threads(16)
    main()
