# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import numpy as np
import importlib
import logging


def print_composite(data, beg=""):
    if isinstance(data, dict):
        print(f'{beg} dict, size = {len(data)}')
        for key, value in data.items():
            print(f'  {beg}{key}:')
            print_composite(value, beg + "    ")
    elif isinstance(data, list):
        print(f'{beg} list, len = {len(data)}')
        for i, item in enumerate(data):
            print(f'  {beg}item {i}')
            print_composite(item, beg + "    ")
    elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        print(f'{beg} array of size {data.shape}')
    else:
        print(f'{beg} {data}')


def list_to_array(l):
    if isinstance(l, list):
        return np.stack([list_to_array(x) for x in l], axis=0)
    elif isinstance(l, str):
        return np.array(float(l))
    elif isinstance(l, float):
        return np.array(l)


def set_logging_format():
    importlib.reload(logging)
    FORMAT = '[%(filename)s:%(lineno)d %(funcName)s()] %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)


def set_logging_file(file_path):
    FORMAT = '[%(filename)s:%(lineno)d %(funcName)s()] %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT,
                        handlers=[
                            logging.FileHandler(file_path),
                            logging.StreamHandler()
                        ])


def set_seed(random_seed):
    import torch, random
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

