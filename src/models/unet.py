"""
Defines the baseline 3D U-Net model using MONAI's official UNet implementation.
"""
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def get_baseline_unet(in_channels=4, out_channels=4):
    """
    in_channels=4  -> t1c, t1n, t2f, t2w stacked
    out_channels=4 -> background, necrotic core, oedema, enhancing tumour
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE,
        dropout=0.1,
    )
    return model