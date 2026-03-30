from typing import Literal

import torch
from einops import reduce
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss

def l1(pred: Tensor, gt: Tensor) -> Tensor:
    loss = l1_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss

def mean_squared_error(pred: Tensor, gt: Tensor,mean: bool=True) -> Tensor:
    loss = mse_loss(pred, gt, reduction="none")
    if mean:
        loss = reduce(loss, "b ... -> b", "mean")
    else:
        loss=loss.mean(dim=-1)
    return loss

def get_loss(type: Literal["mse", "mse_sum", "l1", "emd"]) -> callable:  #"loss_type", "mse"
    """

    Args:
        type (Literal["mse", "mse_sum", "l1", "emd"]): The type of loss to get.

    Returns:
        callable: The loss function.
    """
    if type == "mse": #111
        print("Diffusion_use_mse_loss")
        return mean_squared_error
