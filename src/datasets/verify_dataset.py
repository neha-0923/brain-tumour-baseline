"""
Verifies structural integrity of the BraTS 2023 GLI training dataset.
Checks: correct file count per patient, non-empty files, consistent
volume shapes, and readable NIfTI headers.
"""
import os
import yaml
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def load_config(config_path="configs/baseline_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def verify_dataset(raw_data_dir, expected_modalities, expected_seg_suffix):
    raw_data_dir = Path(raw_data_dir)
    patient_dirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])

    print(f"Found {len(patient_dirs)} patient folders.\n")

    missing_files = []
    empty_files = []
    shape_mismatches = []
    unreadable_files = []
    reference_shape = None

    for patient_dir in tqdm(patient_dirs, desc="Verifying patients"):
        patient_id = patient_dir.name
        expected_files = {
            mod: patient_dir / f"{patient_id}-{mod}.nii.gz"
            for mod in expected_modalities
        }
        expected_files["seg"] = patient_dir / f"{patient_id}-{expected_seg_suffix}.nii.gz"

        volumes_this_patient = {}

        for key, file_path in expected_files.items():
            if not file_path.exists():
                missing_files.append(str(file_path))
                continue
            if file_path.stat().st_size == 0:
                empty_files.append(str(file_path))
                continue
            try:
                img = nib.load(str(file_path))
                shape = img.shape
                volumes_this_patient[key] = shape
                if reference_shape is None:
                    reference_shape = shape
                elif shape != reference_shape:
                    shape_mismatches.append((str(file_path), shape, reference_shape))
            except Exception as e:
                unreadable_files.append((str(file_path), str(e)))

    # ---- Summary report ----
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total patient folders checked : {len(patient_dirs)}")
    print(f"Missing files                 : {len(missing_files)}")
    print(f"Empty (0-byte) files          : {len(empty_files)}")
    print(f"Unreadable / corrupt files    : {len(unreadable_files)}")
    print(f"Shape mismatches              : {len(shape_mismatches)}")
    print(f"Reference volume shape        : {reference_shape}")
    print("=" * 60)

    if missing_files:
        print("\nFirst 10 missing files:")
        for f in missing_files[:10]:
            print(f"  - {f}")

    if empty_files:
        print("\nFirst 10 empty files:")
        for f in empty_files[:10]:
            print(f"  - {f}")

    if unreadable_files:
        print("\nFirst 10 unreadable files:")
        for f, err in unreadable_files[:10]:
            print(f"  - {f} ({err})")

    if shape_mismatches:
        print("\nFirst 10 shape mismatches:")
        for f, shape, ref in shape_mismatches[:10]:
            print(f"  - {f}: {shape} (expected {ref})")

    is_clean = not (missing_files or empty_files or unreadable_files or shape_mismatches)
    print(f"\nDATASET STATUS: {'CLEAN ✅' if is_clean else 'ISSUES FOUND ⚠️'}")

    return {
        "total_patients": len(patient_dirs),
        "missing_files": missing_files,
        "empty_files": empty_files,
        "unreadable_files": unreadable_files,
        "shape_mismatches": shape_mismatches,
        "reference_shape": reference_shape,
        "is_clean": is_clean,
    }

if __name__ == "__main__":
    config = load_config()
    verify_dataset(
        raw_data_dir=config["data"]["raw_data_dir"],
        expected_modalities=config["data"]["expected_modalities"],
        expected_seg_suffix=config["data"]["expected_seg_suffix"],
    )