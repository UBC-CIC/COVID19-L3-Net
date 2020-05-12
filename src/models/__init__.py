# from . import semseg_cost
import torch
import os
import tqdm 
from . import semseg
import torch


def get_model(model_dict, exp_dict=None, train_set=None):
    if model_dict['name'] in ["semseg"]:
        model =  semseg.SemSeg(exp_dict, train_set)

        # load pretrained
        if 'pretrained' in model_dict:
            model.load_state_dict(torch.load(model_dict['pretrained']))
 
    return model





