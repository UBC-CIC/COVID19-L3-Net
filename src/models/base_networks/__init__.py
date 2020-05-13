from . import unet2d
from . import segmentation_models_pytorch as smp


def get_base(base_name, exp_dict, n_classes):
    if base_name == "fcn8_vgg16":
        base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes)

    if base_name == "unet2d":
        base = unet2d.UNet(n_channels=1, n_classes=n_classes)

    if base_name == 'pspnet':
        kwargs = {'encoder_name': exp_dict['model']['encoder'],
                  'in_channels': exp_dict['num_channels'],
                  'encoder_weights': None,  # ignore error. it still works.
                  'classes': n_classes}
        if exp_dict['model']['base'] == 'pspnet':
            net_fn = smp.PSPNet

        assert net_fn is not None

        base = smp.PSPNet(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                          encoder_weights=None,  # ignore error. it still works.
                          classes=n_classes)

    return base
