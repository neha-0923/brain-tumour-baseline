"""
Creates a reproducible, patient-level train/val/test split for BraTS GLI data.
Groups multiple timepoints of the same patient together to avoid data leakage.
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
import yaml

RANDOM_SEED = 42

def get_patient_group(folder_name):
    """
    Extracts the core patient ID (e.g. '00058') from a folder name like
    'BraTS-GLI-00058-000', so that different timepoints of the same
    patient are grouped together.
    """
    match = re.match(r"BraTS-GLI-(\d+)-\d+", folder_name)
    if match:
        return match.group(1)
    raise ValueError(f"Unexpected folder name format: {folder_name}")

def create_split(raw_data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    raw_data_dir = Path(raw_data_dir)
    all_folders = sorted([d.name for d in raw_data_dir.iterdir() if d.is_dir()])

    # Group folder names by underlying patient ID
    patient_to_folders = defaultdict(list)
    for folder in all_folders:
        patient_id = get_patient_group(folder)
        patient_to_folders[patient_id].append(folder)

    unique_patients = sorted(patient_to_folders.keys())
    print(f"Total folders      : {len(all_folders)}")
    print(f"Unique patients    : {len(unique_patients)}")

    # First split off the test set
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=test_ratio, random_state=RANDOM_SEED
    )
    # Then split remaining into train/val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=val_size_adjusted, random_state=RANDOM_SEED
    )

    def expand(patient_list):
        folders = []
        for p in patient_list:
            folders.extend(patient_to_folders[p])
        return sorted(folders)

    split = {
        "train": expand(train_patients),
        "val": expand(val_patients),
        "test": expand(test_patients),
    }

    print(f"\nTrain patients: {len(train_patients)} -> {len(split['train'])} folders")
    print(f"Val patients  : {len(val_patients)} -> {len(split['val'])} folders")
    print(f"Test patients : {len(test_patients)} -> {len(split['test'])} folders")

    # Sanity check: no patient ID appears in more than one split
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)
    assert not (train_set & val_set), "Leakage between train and val!"
    assert not (train_set & test_set), "Leakage between train and test!"
    assert not (val_set & test_set), "Leakage between val and test!"
    print("\nNo patient-level leakage between splits. ✅")

    return split

if __name__ == "__main__":
    with open("configs/baseline_config.yaml") as f:
        config = yaml.safe_load(f)

    split = create_split(config["data"]["raw_data_dir"])

    output_dir = Path("outputs/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_val_test_split.json"

    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit saved to {output_path}")