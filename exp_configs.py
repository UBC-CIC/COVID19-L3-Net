from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}


EXP_GROUPS["open_source_unet2d"] = hu.cartesian_exp_group({
        'batch_size': 1,
        'num_channels':1,
        'dataset': [{
                'name':'open_source', 
                'transform':'basic', 
                'transform_mode':3
        }],
        'dataset_size':[{
                'train':'all', 
                'val':'all'
        }],
        'max_epoch': [100],
        'optimizer': ["adam"], 
        'lr': [1e-5,],
        'model': [{
                'name':'semseg', 
                'base':'unet2d', 
                'loss':'dice', 
                'pretrained':'checkpoints/unet2d_aio_pre_luna.ckpt'
        }],
})

EXP_GROUPS["open_source_pspnet"] = hu.cartesian_exp_group({
        'batch_size': 1,
        'num_channels':1,
        'dataset': [{
                'name':'open_source', 
                'transform':'basic', 
                'transform_mode':3
        }],
        'dataset_size':[{
                'train':'all', 
                'val':'all'
        }],
        'max_epoch': [100],
        'optimizer': ["adam"], 
        'lr': [ 1e-5,],
        'model': [{
                'name':'semseg', 
                'encoder':'inceptionresnetv2', 
                'base':'pspnet', 
                'loss':'dice'
        }]
})