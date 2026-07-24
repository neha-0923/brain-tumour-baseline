"""
Sanity-checks the optimiser + scheduler: runs a few dummy training steps
on a small random batch and confirms (a) loss decreases, (b) LR follows
the expected cosine decay pattern.
"""
import torch
from src.models.unet import get_baseline_unet
from src.training.losses import get_baseline_loss
from src.training.optimiser import get_optimizer_and_scheduler

if __name__ == "__main__":
    torch.manual_seed(42)

    model = get_baseline_unet(in_channels=4, out_channels=4)
    loss_fn = get_baseline_loss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, initial_lr=2e-4, max_epochs=10)

    # Small dummy patch to keep this fast on CPU (not real data, just for
    # confirming the optimiser mechanics work correctly)
    dummy_image = torch.randn(1, 4, 32, 32, 32)
    dummy_seg = torch.randint(0, 4, (1, 1, 32, 32, 32)).float()

    model.train()
    print("Running 5 dummy training steps...\n")
    for step in range(5):
        optimizer.zero_grad()
        logits = model(dummy_image)
        loss = loss_fn(logits, dummy_seg)
        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Step {step+1}: loss={loss.item():.4f}, lr={current_lr:.6f}")

    print("\nStepping scheduler forward through remaining epochs to confirm cosine decay:")
    for epoch in range(10):
        scheduler.step()
        print(f"  epoch {epoch+1}: lr={optimizer.param_groups[0]['lr']:.6f}")