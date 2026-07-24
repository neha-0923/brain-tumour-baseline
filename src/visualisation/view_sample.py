"""
Loads one BraTS patient's 4 modalities + segmentation mask,
prints basic info, and displays a mid-axial slice of each.
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def load_patient_volumes(patient_dir, modalities, seg_suffix):
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name
    volumes = {}
    for mod in modalities:
        path = patient_dir / f"{patient_id}-{mod}.nii.gz"
        volumes[mod] = nib.load(str(path)).get_fdata()
    seg_path = patient_dir / f"{patient_id}-{seg_suffix}.nii.gz"
    volumes["seg"] = nib.load(str(seg_path)).get_fdata()
    return volumes

def print_volume_info(volumes):
    for name, vol in volumes.items():
        print(f"{name:5s} | shape={vol.shape} | dtype={vol.dtype} | "
              f"min={vol.min():.1f} max={vol.max():.1f} unique_labels={np.unique(vol) if name=='seg' else ''}")

def show_mid_slice(volumes, slice_axis=2):
    modalities = [k for k in volumes.keys() if k != "seg"]
    mid_idx = volumes[modalities[0]].shape[slice_axis] // 2

    fig, axes = plt.subplots(1, len(modalities) + 1, figsize=(15, 4))

    for i, mod in enumerate(modalities):
        vol = volumes[mod]
        sl = np.take(vol, mid_idx, axis=slice_axis)
        axes[i].imshow(sl.T, cmap="gray", origin="lower")
        axes[i].set_title(mod)
        axes[i].axis("off")

    seg_slice = np.take(volumes["seg"], mid_idx, axis=slice_axis)
    axes[-1].imshow(seg_slice.T, cmap="viridis", origin="lower")
    axes[-1].set_title("Segmentation (labels 0-3)")
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/figures/sample_slice_visualisation.png", dpi=150)
    print("\nSaved figure to outputs/figures/sample_slice_visualisation.png")
    plt.show()

if __name__ == "__main__":
    with open("configs/baseline_config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data"]["raw_data_dir"])
    example_patient_dir = raw_dir / "BraTS-GLI-00058-000"  # the one we already verified

    volumes = load_patient_volumes(
        example_patient_dir,
        config["data"]["expected_modalities"],
        config["data"]["expected_seg_suffix"]
    )
    print_volume_info(volumes)
    show_mid_slice(volumes)