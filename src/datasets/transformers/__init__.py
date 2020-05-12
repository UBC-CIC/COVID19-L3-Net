import torch
import numpy as np
import random

from scipy.ndimage import zoom
from torchvision import transforms
from . import trans_utils as tu
from haven import haven_utils as hu
# from batchgenerators.augmentations import crop_and_pad_augmentations
from . import micnn_augmentor


def apply_transform(split, image, label=None, transform_name='basic',
                    exp_dict=None):

    if transform_name == 'basic':
        image /= 4095

        if exp_dict['dataset']['transform_mode'] is not None:
            image, label = image[200:550], label[200:550]

        if split == 'train':
            if exp_dict['dataset']['transform_mode'] == 2:
                da = micnn_augmentor.Data_Augmentation()
                image, label = da.run(image[None,None], label[None,None])
                
            if exp_dict['dataset']['transform_mode'] == 3:
                if np.random.rand() < 0.5:
                    da = micnn_augmentor.Data_Augmentation()
                    image, label = da.run(image[None,None], label[None,None])

        image = image.squeeze()
        label = label.squeeze()
        # hu.save_image('tmp_after.png', image)
        
        image = torch.FloatTensor(image)[None]
        # assert image.min() >= 0
        # assert image.max() <= 1
        
        normalize = transforms.Normalize((0.5,), (0.5,))
        image = normalize(image)
        return image, label

    if transform_name == 'basic_hu':
        image+= 1024
        image /= 5024
        assert image.min()>=0 and image.max() <= 1
        # if exp_dict['dataset']['transform_mode'] is not None:
        #     image, label = image[200:550], label[200:550]
        class_map = tu.get_class_map(exp_dict['n_classes'])
        lbl_trans = transforms.Compose([
            tu.GroupLabels(class_map),
            tu.PreparePilLabel(),
            transforms.ToPILImage(),
            tu.UndoPreparePilLabel(),
            transforms.ToTensor(),
            tu.Squeeze(),
        ])
        label = lbl_trans(label)

        if split == 'train':
            if exp_dict['dataset']['transform_mode'] == 2:
                da = micnn_augmentor.Data_Augmentation()
                image, label = da.run(image[None,None], label[None,None])
                
            if exp_dict['dataset']['transform_mode'] == 3:
                if np.random.rand() < 0.5:
                    da = micnn_augmentor.Data_Augmentation()
                    image, label = da.run(image[None,None], label[None,None].numpy())
                    label = torch.LongTensor(label).squeeze()

        image = image.squeeze()
        label = label.squeeze()
        # hu.save_image('tmp_after.png', image)
        
        image = torch.FloatTensor(image)[None]
        # assert image.min() >= 0
        # assert image.max() <= 1
        
        normalize = transforms.Normalize((0.5,), (0.5,))
        image = normalize(image)
        return image, label

    elif transform_name == 'mdai_basic':
        img_trans = transforms.Compose([
            tu.Threshold(min=-1000, max=50),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor([-653.2204]),
                std=torch.tensor([628.5188])
            )
        ])
        class_map = tu.get_class_map(exp_dict['n_classes'])
        lbl_trans = transforms.Compose([
            tu.GroupLabels(class_map),
            tu.PreparePilLabel(),
            transforms.ToPILImage(),
            tu.UndoPreparePilLabel(),
            transforms.ToTensor(),
            tu.Squeeze(),
        ])
        image, label = img_trans(image), lbl_trans(label)
        if split == 'train':
            if exp_dict['dataset'].get('transform_mode') == 3:
                if np.random.rand() < 0.5:
                    da = micnn_augmentor.Data_Augmentation()
                    image, label = da.run(image.numpy()[None], label.numpy()[None,None])
                    image, label = torch.FloatTensor(image).squeeze()[None], torch.LongTensor(label).squeeze()
        if label is None:
            return image
        return image, label

    elif transform_name == 'pspnet_transformer':
        windows = ['lung']
        mean, std = tu.get_normalization_stats(windows)
        thresh_min, thresh_max = tu.get_thresholds_stats(windows)

        img_trans = transforms.Compose([
            tu.Threshold(min=thresh_min, max=thresh_max),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ])
        
        class_map = tu.get_class_map(exp_dict['n_classes'])
        lbl_trans = transforms.Compose([
            tu.GroupLabels(class_map),
            tu.PreparePilLabel(),
            transforms.ToPILImage(),
            tu.UndoPreparePilLabel(),
            transforms.ToTensor(),
            tu.Squeeze(),
        ])
        if label is None:
            return img_trans(image)
        return img_trans(image).float(), lbl_trans(label)


