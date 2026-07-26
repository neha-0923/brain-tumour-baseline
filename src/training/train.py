"""
Full baseline U-Net training loop with:
- CacheDataset (cache_rate=0.5) for speed
- persistent_workers to avoid Colab notebook hangs
- checkpoint + history saved to Google Drive every epoch (survives disconnects)
- resume-from-checkpoint support
"""
import yaml
import json
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import list_data_collate

from src.preprocessing.transforms import get_train_transforms, get_val_transforms
from src.datasets.brats_dataset import build_cache_dataset
from src.models.unet import get_baseline_unet
from src.training.losses import get_baseline_loss
from src.training.optimiser import get_optimizer_and_scheduler


def train_baseline(config_path="configs/baseline_config.yaml", max_epochs=100, batch_size=2,
                    drive_output_dir="/content/drive/MyDrive/brain_tumour_baseline_outputs"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    modalities = config["data"]["expected_modalities"]
    seg_suffix = config["data"]["expected_seg_suffix"]
    raw_dir = config["data"]["raw_data_dir"]
    split_file = "outputs/splits/train_val_test_split.json"

    # ---- Drive-backed output paths (survive Colab disconnects) ----
    checkpoint_dir = Path(drive_output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(drive_output_dir) / "training_history.json"
    last_checkpoint_path = checkpoint_dir / "last_checkpoint.pth"
    best_checkpoint_path = checkpoint_dir / "best_model.pth"

    print("Building training CacheDataset (first run takes a while - "
          "caching 50% of patients' preprocessed volumes in RAM)...")
    train_ds = build_cache_dataset(
        raw_dir, split_file, "train", modalities, seg_suffix,
        get_train_transforms(modalities, patch_size=(128, 128, 128)),
        cache_rate=0.5, num_workers=2,
    )
    print("Building validation CacheDataset...")
    val_ds = build_cache_dataset(
        raw_dir, split_file, "val", modalities, seg_suffix,
        get_val_transforms(modalities),
        cache_rate=0.5, num_workers=2,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=list_data_collate,
        persistent_workers=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2,
        persistent_workers=True, pin_memory=True,
    )

    model = get_baseline_unet(in_channels=4, out_channels=4).to(device)
    loss_fn = get_baseline_loss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, initial_lr=2e-4, max_epochs=max_epochs)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    start_epoch = 0
    best_val_dice = -1.0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    # ---- Resume from checkpoint if one exists on Drive ----
    if last_checkpoint_path.exists():
        print(f"\nFound existing checkpoint at {last_checkpoint_path} — resuming.")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint["best_val_dice"]
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
        print(f"Resuming from epoch {start_epoch}, best_val_dice so far: {best_val_dice:.4f}\n")
    else:
        print("\nNo existing checkpoint found — starting fresh from epoch 0.\n")

    for epoch in range(start_epoch, max_epochs):
        # ---- Training phase ----
        model.train()
        epoch_train_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} training")):
            images = batch["image"].to(device)
            labels = batch["seg"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        scheduler.step()

        # ---- Validation phase ----
        model.eval()
        epoch_val_loss = 0.0
        dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} validation"):
                images = batch["image"].to(device)
                labels = batch["seg"].to(device)

                outputs = sliding_window_inference(
                    images, roi_size=(128, 128, 128), sw_batch_size=1, predictor=model,
                )
                val_loss = loss_fn(outputs, labels)
                epoch_val_loss += val_loss.item()

                preds_list = [post_pred(o) for o in torch.unbind(outputs, dim=0)]
                labels_list = [post_label(l) for l in torch.unbind(labels, dim=0)]
                dice_metric(y_pred=preds_list, y=labels_list)

        epoch_val_loss /= len(val_loader)
        epoch_val_dice = dice_metric.aggregate().item()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_dice"].append(epoch_val_dice)

        print(f"Epoch {epoch+1}/{max_epochs} | "
              f"train_loss={epoch_train_loss:.4f} | "
              f"val_loss={epoch_val_loss:.4f} | "
              f"val_dice={epoch_val_dice:.4f}")

        # ---- Save history to Drive every epoch ----
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # ---- Save "last checkpoint" every epoch (for resuming) ----
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
        }, last_checkpoint_path)

        # ---- Save "best model" separately when val_dice improves ----
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> New best model saved (val_dice={best_val_dice:.4f})")

    return history


if __name__ == "__main__":
    train_baseline(max_epochs=100, batch_size=2)