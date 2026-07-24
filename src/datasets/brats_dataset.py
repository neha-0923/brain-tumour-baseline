"""
PyTorch Dataset wrapping the BraTS split JSON and MONAI transform pipelines.
"""
import json
from pathlib import Path
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, raw_data_dir, split_file, split_name, modalities, seg_suffix, transforms):
        self.raw_data_dir = Path(raw_data_dir)
        self.modalities = modalities
        self.seg_suffix = seg_suffix
        self.transforms = transforms

        with open(split_file, "r") as f:
            split = json.load(f)
        self.patient_folders = split[split_name]

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        folder_name = self.patient_folders[idx]
        patient_dir = self.raw_data_dir / folder_name

        data_dict = {
            mod: str(patient_dir / f"{folder_name}-{mod}.nii.gz")
            for mod in self.modalities
        }
        data_dict["seg"] = str(patient_dir / f"{folder_name}-{self.seg_suffix}.nii.gz")

        result = self.transforms(data_dict)

        # Training transforms (RandCropByPosNegLabeld) return a list of dicts
        # even with num_samples=1; validation transforms return a single dict.
        if isinstance(result, list):
            result = result[0]

        return {"image": result["image"], "seg": result["seg"]}