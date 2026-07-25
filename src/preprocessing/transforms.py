"""
Defines the MONAI transform pipelines for BraTS preprocessing.
Includes: loading, normalisation, cropping (shared by train/val/test),
and training-only augmentation (random patch cropping + flips/noise).
"""
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandShiftIntensityd,
)

def get_base_transforms(modalities, seg_key="seg"):
    """
    Shared preprocessing steps used for ALL splits (train/val/test):
    loading, channel-first formatting, normalisation, stacking, and
    foreground cropping. No randomness here — deterministic for every patient.
    """
    keys = modalities + [seg_key]

    transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        NormalizeIntensityd(keys=modalities, nonzero=True, channel_wise=True),
        ConcatItemsd(keys=modalities, name="image", dim=0),
        CropForegroundd(
            keys=["image", seg_key],
            source_key="image",
            margin=5,
            allow_smaller=False,
        ),
    ])
    return transforms

def get_train_transforms(modalities, seg_key="seg", patch_size=(128, 128, 128)):
    """
    Training-only pipeline: base preprocessing + guaranteed minimum-size
    padding + random positive/negative patch cropping + augmentation.
    """
    base = get_base_transforms(modalities, seg_key).transforms

    augment = [
        SpatialPadd(
            keys=["image", seg_key],
            spatial_size=patch_size,
            mode="constant",
        ),
        RandCropByPosNegLabeld(
            keys=["image", seg_key],
            label_key=seg_key,
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", seg_key], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", seg_key], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", seg_key], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", seg_key], prob=0.5, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
    ]

    return Compose(list(base) + augment)

def get_val_transforms(modalities, seg_key="seg"):
    """
    Validation/test pipeline: base preprocessing only, no randomness,
    no augmentation, no fixed-size cropping (evaluated on full cropped volume).
    """
    return get_base_transforms(modalities, seg_key)