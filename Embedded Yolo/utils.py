import random
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_init_seeds(seed=seed)


def torch_init_seeds(seed=0):
    torch.manual_seed(seed)

    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
