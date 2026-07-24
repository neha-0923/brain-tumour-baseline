"""
Sanity-checks the full training transform pipeline (normalisation,
cropping, random patch extraction, and augmentation) on one real patient.
"""
import yaml
from pathlib import Path
from src.preprocessing.transforms import get_train_transforms, get_val_transforms

if __name__ == "__main__":
    with open("configs/baseline_config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data"]["raw_data_dir"])
    modalities = config["data"]["expected_modalities"]
    seg_suffix = config["data"]["expected_seg_suffix"]

    patient_dir = raw_dir / "BraTS-GLI-00058-000"
    patient_id = patient_dir.name

    data_dict = {
        mod: str(patient_dir / f"{patient_id}-{mod}.nii.gz") for mod in modalities
    }
    data_dict["seg"] = str(patient_dir / f"{patient_id}-{seg_suffix}.nii.gz")

    print("=" * 60)
    print("TRAINING TRANSFORMS (with augmentation + patch cropping)")
    print("=" * 60)
    train_transforms = get_train_transforms(modalities, patch_size=(128, 128, 128))
    train_result = train_transforms(data_dict)

    # RandCropByPosNegLabeld with num_samples=1 returns a list of length 1
    sample = train_result[0]
    print(f"Patch image shape: {sample['image'].shape}")   # expect (4, 128, 128, 128)
    print(f"Patch seg shape  : {sample['seg'].shape}")     # expect (1, 128, 128, 128)
    import torch
    print(f"Unique labels in this patch: {torch.unique(sample['seg'])}")

    print("\n" + "=" * 60)
    print("VALIDATION TRANSFORMS (no augmentation, full cropped volume)")
    print("=" * 60)
    val_transforms = get_val_transforms(modalities)
    val_result = val_transforms(data_dict)
    print(f"Val image shape: {val_result['image'].shape}")
    print(f"Val seg shape  : {val_result['seg'].shape}")