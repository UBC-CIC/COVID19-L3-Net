import torchvision
import torch
import numpy as np
from torchvision.transforms import transforms
from sklearn.utils import shuffle
from PIL import Image
from . import open_source
from src import utils as ut
import os
import os
import numpy as np

import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


def get_dataset(dataset_dict, split, datadir, exp_dict, dataset_size=None):
    name = dataset_dict['name']

    if name == "open_source":
        dataset = open_source.OpenSource(split=split, datadir=datadir, exp_dict=exp_dict)
        if dataset_size is not None and dataset_size[split] != 'all':
            dataset.img_tgt_dict = dataset.img_tgt_dict[25:25+dataset_size[split]]

        return dataset

    elif name == "mdai":
        dataset = mdai.Mdai(split=split, datadir=datadir, exp_dict=exp_dict)
        if dataset_size is not None and dataset_size[split] != 'all':
            dataset.active_data = dataset.active_data[:dataset_size[split]]

        return dataset

    elif name == "combined":
        dataset = combined.Combined(split=split, datadir=datadir, exp_dict=exp_dict)

        return dataset

    else:
        raise ValueError('dataset %s not found' % name)
