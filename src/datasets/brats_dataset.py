"""
Builds MONAI CacheDataset objects wrapping the BraTS split JSON.
CacheDataset caches deterministic preprocessing (load/normalize/crop/pad)
in RAM, only re-running random augmentation fresh each epoch — this
avoids re-reading and re-processing NIfTI files from disk every epoch.
"""
import json
from pathlib import Path
from monai.data import CacheDataset

def build_data_list(raw_data_dir, split_file, split_name, modalities, seg_suffix):
    """
    Builds a list of {modality: filepath, ...} dicts for the given split —
    the raw input format CacheDataset expects, before any transform runs.
    """
    raw_data_dir = Path(raw_data_dir)
    with open(split_file, "r") as f:
        split = json.load(f)

    data_list = []
    for folder_name in split[split_name]:
        patient_dir = raw_data_dir / folder_name
        data_dict = {
            mod: str(patient_dir / f"{folder_name}-{mod}.nii.gz")
            for mod in modalities
        }
        data_dict["seg"] = str(patient_dir / f"{folder_name}-{seg_suffix}.nii.gz")
        data_list.append(data_dict)

    return data_list

def build_cache_dataset(raw_data_dir, split_file, split_name, modalities, seg_suffix,
                         transforms, cache_rate=0.5, num_workers=2):
    data_list = build_data_list(raw_data_dir, split_file, split_name, modalities, seg_suffix)
    return CacheDataset(
        data=data_list,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )