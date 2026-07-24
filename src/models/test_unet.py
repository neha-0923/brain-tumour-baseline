"""
Sanity-checks the baseline U-Net: builds it, runs a dummy forward pass
with the exact patch size used in training, and reports parameter count.
"""
import torch
from src.models.unet import get_baseline_unet

if __name__ == "__main__":
    model = get_baseline_unet(in_channels=4, out_channels=4)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    dummy_input = torch.randn(1, 4, 128, 128, 128)  # (batch, channels, D, H, W)
    print(f"\nInput shape : {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # expect (1, 4, 128, 128, 128)

    assert output.shape[2:] == dummy_input.shape[2:], "Spatial dimensions mismatch!"
    assert output.shape[1] == 4, "Output channel count mismatch!"
    print("\nForward pass shape check passed. ")