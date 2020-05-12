from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets
from src import utils as ut


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy, shutil
cudnn.benchmark = True



def test(exp_dict, savedir_base, datadir,  num_workers=0, model_path=None, scan_id=None):
    # bookkeepting stuff
    # ==================
    pprint.pprint(exp_dict)
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    print("Experiment saved in %s" % savedir)

    # Dataset
    # ==================
    # val set
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="val",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])
    if str(scan_id) != 'None':
        test_set.active_data = test_set.get_scan(scan_id)
    test_sampler = torch.utils.data.SequentialSampler(test_set)
    test_loader = DataLoader(test_set,
                            sampler=test_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=num_workers)

    # Model
    # ==================
    # chk = torch.load('best_model.ckpt')
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=test_set).cuda()
    epoch = -1

    if str(model_path) != 'None':
        model_path = model_path
        model.load_state_dict(hu.torch_load(model_path))
    else:
        try:
            exp_dict_train = copy.deepcopy(exp_dict)
            del exp_dict_train['test_mode']
            savedir_train = os.path.join(savedir_base, hu.hash_dict(exp_dict_train))
            model_path = os.path.join(savedir_train, "model_best.pth")
            score_list = hu.load_pkl(os.path.join(savedir_train, 'score_list_best.pkl'))
            epoch = score_list[-1]['epoch']
            print('Loaded model at epoch %d with score %.3f' % epoch)
            model.load_state_dict(hu.torch_load(model_path))
        except:
            pass

        


    s_time = time.time()
    savedir_images = os.path.join(savedir, 'images')
    
    # delete image folder if exists
    if os.path.exists(savedir_images):
        shutil.rmtree(savedir_images)
        
    os.makedirs(savedir_images, exist_ok=True)
    # for i in range(20):
    #     score_dict = model.train_on_loader(test_loader)
    score_dict = model.val_on_loader(test_loader,
                                    savedir_images=savedir_images,
                                    n_images=30000, save_preds=True)

    score_dict['epoch'] = epoch
    score_dict["time"] = time.time() - s_time
    score_dict["saved_at"] = hu.time_to_montreal()
    # save test_score_list
    test_path = os.path.join(savedir, "score_list.pkl")
    if os.path.exists(test_path):
        test_score_list = [sd for sd in hu.load_pkl(test_path) if sd['epoch'] != epoch]
    else:
        test_score_list = []

    # append score_dict to last result
    test_score_list += [score_dict]
    hu.save_pkl(test_path, test_score_list)
    print('Final Score is ', str(score_dict["val_score"]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=None)
    parser.add_argument("-p", "--model_path", default=None)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-si", "--scan_id", type=str, default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

        # format them for test
        for exp_dict in exp_list:
            exp_dict['test_mode'] = 1

    # Run experiments or View them
    # ----------------------------
    if args.run_jobs:
        # launch jobs
        import usr_configs as uc
        run_command = hu.create_command('python test.py -ei <exp_id>', args)
        uc.run_jobs(exp_list, args.savedir_base, args.datadir)

    else:
        # test experiments
        for exp_dict in exp_list:
            # do trainval
            test(exp_dict=exp_dict,
                savedir_base=args.savedir_base,
                datadir=args.datadir,
                num_workers=args.num_workers,
                model_path=args.model_path,
                scan_id=args.scan_id)
