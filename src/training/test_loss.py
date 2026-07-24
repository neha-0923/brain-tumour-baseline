"""
Sanity-checks the loss function using the real U-Net + a real preprocessed
patient patch, confirming the loss computes correctly and produces a
sensible scalar value.
"""
import yaml
import torch
from pathlib import Path
from src.preprocessing.transforms import get_train_transforms
from src.models.unet import get_baseline_unet
from src.training.losses import get_baseline_loss

if __name__ == "__main__":
    with open("configs/baseline_config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data"]["raw_data_dir"])
    modalities = config["data"]["expected_modalities"]
    seg_suffix = config["data"]["expected_seg_suffix"]

    patient_dir = raw_dir / "BraTS-GLI-00058-000"
    patient_id = patient_dir.name
    data_dict = {mod: str(patient_dir / f"{patient_id}-{mod}.nii.gz") for mod in modalities}
    data_dict["seg"] = str(patient_dir / f"{patient_id}-{seg_suffix}.nii.gz")

    transforms = get_train_transforms(modalities, patch_size=(128, 128, 128))
    sample = transforms(data_dict)[0]

    image = sample["image"].unsqueeze(0)  # add batch dim -> (1, 4, 128, 128, 128)
    seg = sample["seg"].unsqueeze(0)       # -> (1, 1, 128, 128, 128)

    print(f"Image batch shape: {image.shape}")
    print(f"Seg batch shape  : {seg.shape}")

    model = get_baseline_unet(in_channels=4, out_channels=4)
    loss_fn = get_baseline_loss()

    model.eval()
    with torch.no_grad():
        logits = model(image)  # (1, 4, 128, 128, 128)
        loss_value = loss_fn(logits, seg)

    print(f"\nModel output (logits) shape: {logits.shape}")
    print(f"Loss value (untrained model): {loss_value.item():.4f}")
    print("\nSanity check: loss should be a single positive finite scalar.")
    assert torch.isfinite(loss_value), "Loss is NaN or Inf!"
    print("Loss is finite. ")