import torch
import numpy as np
import random

from scipy.ndimage import zoom
from torchvision import transforms

def get_class_map(n_classes):

    if n_classes==5:
        class_map = {
                -1:-1,
                0:0,
                1:1,
                2:1,
                3:-1,
                4:-1,
                5:2,
                6:3,
                7:3,
                8:4,
                9:4,
                10:4,
            }

    elif n_classes == 3:
        class_map = {
                -1:-1,
                0:0,
                1:1,
                2:1,
                3:-1,
                4:-1,
                5:2,
                6:2,
                7:2,
                8:2,
                9:2,
                10:2,
            }
    return class_map
def get_thresholds_stats(windows):
    # Lung level and width
    window_level = 0
    window_width = 3000
    thresh_max = []
    thresh_min = []
    for window in windows:
        if window == 'emphysema':
            window_level = -800
            window_width = 800
        if window == 'lung':
            window_level = -500
            window_width = 1500
        if window == 'mediastinum':
            window_level = 40
            window_width = 400
        thresh_max.append(window_level + window_width / 2)
        thresh_min.append(window_level - window_width / 2)
    return thresh_min, thresh_max


def get_normalization_stats(windows):
    window_index = {'lung': 0,
                    'emphysema': 1,
                    'mediastinum': 2}
    mean = torch.tensor([-580.1847, -733.5521, -93.1991])
    std = torch.tensor([478.6256, 281.3584, 117.1486])
    out_mean = []
    out_std = []
    for window in windows:
        out_mean.append(torch.tensor([mean[window_index[window]]]))
        out_std.append(torch.tensor([std[window_index[window]]]))

    if len(windows) == 1:
        out_mean = torch.tensor([out_mean[0]])
        out_std = torch.tensor([out_std[0]])

    return out_mean, out_std


# transform funcs
# ---------------

class GroupLabels:
    def __init__(self, class_map):
        self.class_map = class_map

    def __call__(self, label):

        assert len(label.shape) == 3, 'Logic only works on 2D labels right now'

        new_label = np.zeros(label.shape[-2:], dtype=np.float)
        new_label.fill(-1)

        # Replace background in group 2 with -1
        label[1, label[1] == 0] = -1

        # If a slice has lung labels, we start from there
        if (label[0] > 0).any():
            new_label = label[0]

        # If there is any pleural effusion, include only the positive
        if (label[1] > 0).any():
            inds = label[1] > 0
            new_label[inds] = label[1, inds]

        # If there are GGO labels, include them
        if (label[2] > 0).any():
            inds = label[2] > 0
            new_label[inds] = label[2, inds]

        # If there are no GGO labels, we need to clear the lungs but can keep the BG
        else:
            new_label[label[0] > 0] = -1

        # Remove Expert annotations
        new_label[new_label > 10] = -1

        # Apply class mapping
        for (class_id, target) in self.class_map.items():
            new_label[new_label == class_id] = target
        return new_label


class RepeatChannels:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        out = x.repeat(self.n, 1, 1, 1)
        return out


class Normalize3D:
    def __init__(self, mean=None, std=None):
        assert isinstance(
            mean, torch.Tensor), 'Input vectors must be a tensor with 1 element per channel'
        assert isinstance(
            std, torch.Tensor), 'Input vectors must be a tensor with 1 element per channel'
        self.mean = None
        self.std = None
        if mean:
            self.mean = mean[:, None, None, None].float()
        if std:
            self.std = std[:, None, None, None].float()

    def __call__(self, x):
        if self.mean:
            x = x - self.mean
        if self.std:
            x = x/self.std
        return x


class RandomDepthCrop:
    def __init__(self, n):
        self.n = n
        self.depth_crop = DepthCrop(self.n)

    def __call__(self, x):
        out = None
        C, D, W, H = x.shape

        if self.n >= D:
            out = self.depth_crop(x)
        else:
            start_ind = random.randint(0, D-self.n)
            idx = np.arange(start_ind, start_ind + self.n)

            if isinstance(x, np.ndarray):
                out = np.take(x, idx, axis=-3)
            elif isinstance(x, torch.Tensor):
                out = torch.index_select(x, -3, torch.from_numpy(idx))
            else:
                raise NotImplementedError()

        if isinstance(out, np.ndarray):
            raise Exception()

        return out


class DepthCrop:
    def __init__(self, n):
        self.n = n
        self.idx = np.arange(0, self.n)

    def __call__(self, x):

        out = None
        D = x.shape[-3]

        pad = 0
        if D < self.n:
            pad = self.n - D

        x_pad = np.pad(x, (
            (0, 0),
            (0, pad),
            (0, 0),
            (0, 0)),
            mode='constant'
        )

        if isinstance(x, np.ndarray):
            out = np.take(x_pad, self.idx, axis=-3)
        elif isinstance(x, torch.Tensor):
            out = torch.index_select(torch.from_numpy(
                x_pad), -3, torch.from_numpy(self.idx))
        else:
            raise NotImplementedError()

        return out


class Threshold:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def __call__(self, x):
        assert isinstance(
            x, np.ndarray), 'Input to threshold must be a np.ndarray'
        x = np.clip(x, self.min, self.max)

        return x


class PreparePilLabel:
    def __call__(self, x):
        return np.uint8(x + 1)


class UndoPreparePilLabel:
    def __call__(self, x):
        return np.int64(x) - 1


class Squeeze:
    def __call__(self, x):
        return x.squeeze()


class ToLabel:
    def __init__(self, group_all=False):
        self.group_all = group_all

    def __call__(self, x):
        if self.group_all:
            x[x > 0] = 1
        x = torch.from_numpy(x).long()
        return x


class ToImage:
    def __call__(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)  # Add channel dim
        return x


class Zoom:
    def __init__(self, zoom, order=2):
        self.zoom = zoom
        self.order = order

    def __call__(self, x):
        x = zoom(x, self.zoom, order=self.order)
        return x


class Resize:
    def __init__(self, new_shape):
        """
        3D linear interpolation

        Arguments:
            new_shape {int, list{int}} -- If `new_shape` is an int, the image is reshaped 
                to be `[new_depth, new_shape, new_shape]`, where new_depth is computed for
                each volume to preserve the aspect ratio. If `new_shape` is a list of three
                `int` values, the image is reshaped accordingly.
        """

        self.calc_new_shape = False
        if isinstance(new_shape, list):
            assert all([isinstance(i, int) for i in new_shape]
                       ), 'New shape must be list of ints'
            self.new_shape = new_shape
        elif isinstance(new_shape, int):
            self.new_width = new_shape
            self.calc_new_shape = True
        else:
            raise Exception('Input must be either an int or a list on ints')

    def __call__(self, x):
        raise NotImplementedError()


class ResizeLinear(Resize):

    def __call__(self, x):
        if self.calc_new_shape:
            old_shape = x.shape
            scale = self.new_width/float(old_shape[1])
            new_shape = [int(np.round(scale*d)) for d in old_shape]
        else:
            new_shape = self.new_shape.copy()
            old_shape = x.shape
            for i in range(len(new_shape)):
                if new_shape[i] == -1:
                    new_shape[i] = x.shape[i]

        x = linearInter3Dreg(x, new_shape)
        return x


class ResizeNN(Resize):

    def __call__(self, x):
        if self.calc_new_shape:
            old_shape = x.shape
            scale = self.new_width/float(old_shape[1])
            new_shape = [int(np.round(scale*d)) for d in old_shape]
        else:
            new_shape = self.new_shape.copy()
            for i in range(len(new_shape)):
                if new_shape[i] == -1:
                    new_shape[i] = x.shape[i]

        x = nnInter3Dreg(x, new_shape)
        return x


class CenterCrop:

    def __init__(self, D, W, H):
        "return CxDxWxH Image, uses zero padding if necessary"
        self.D = D
        self.W = W
        self.H = H

    def __call__(self, x):

        D = self.D
        W = self.W
        H = self.H

        # Crop Depth
        if self.D is not None:
            depth_start = x.shape[-3]//2 - D//2
            if depth_start < 0:
                depth_pad_start = -depth_start
                depth_pad_end = depth_pad_start + x.shape[-3]
                depth_start = 0
            else:
                depth_pad_start = None
                depth_pad_end = None
            depth_end = depth_start + D
        else:
            D = x.shape[-3]
            depth_pad_start = None
            depth_pad_end = None
            depth_start = None
            depth_end = None

        # Crop Width
        if self.W is not None:
            width_start = x.shape[-2]//2 - W//2
            if width_start < 0:
                width_pad_start = -width_start
                width_pad_end = width_pad_start + x.shape[-2]
                width_start = 0
            else:
                width_pad_start = None
                width_pad_end = None
            width_end = width_start + W
        else:
            W = x.shape[-2]
            width_pad_start = None
            width_pad_end = None
            width_start = None
            width_end = None

        # Crop Height
        if self.H is not None:
            height_start = x.shape[-1]//2 - H//2
            if height_start < 0:
                height_pad_start = -height_start
                height_pad_end = height_pad_start + x.shape[-1]
                height_start = 0
            else:
                height_pad_start = None
                height_pad_end = None
            height_end = height_start + H
        else:
            H = x.shape[-1]
            height_pad_start = None
            height_pad_end = None
            height_start = None
            height_end = None

        out = np.zeros(x.shape[:-3] + (D, W, H))
        out[..., depth_pad_start:depth_pad_end, width_pad_start:width_pad_end, height_pad_start:height_pad_end] = x[
            ...,
            depth_start:depth_end,
            width_start:width_end,
            height_start:height_end
        ]

        return out
