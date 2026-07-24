"""
Defines the loss function used for baseline U-Net training.
Uses MONAI's DiceCELoss: a combination of Dice loss (handles class
imbalance, targets the Dice metric directly) and Cross-Entropy loss
(stable gradients, especially early in training).
"""
from monai.losses import DiceCELoss

def get_baseline_loss():
    """
    softmax=True   -> applies softmax internally to model's raw logits
                       before computing loss (our model outputs raw logits,
                       not probabilities, so this must be True)
    to_onehot_y=True -> converts integer segmentation labels (0-3) into
                       one-hot format internally, matching model's
                       4-channel output
    """
    loss_fn = DiceCELoss(
        softmax=True,
        to_onehot_y=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
    return loss_fn