"""
Defines the optimiser and learning rate schedule for baseline U-Net training.
"""
import torch

def get_optimizer_and_scheduler(model, initial_lr=2e-4, weight_decay=1e-5, max_epochs=100):
    """
    AdamW optimiser with cosine-annealed learning rate decay.

    initial_lr    -> starting learning rate
    weight_decay  -> L2 regularisation strength, helps prevent overfitting
    max_epochs    -> total planned training epochs; cosine schedule decays
                     LR to near-zero exactly by this point
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-6,
    )

    return optimizer, scheduler