"""
from: https://github.com/educating-dip/score_based_model_baselines/blob/main/src/utils/losses.py
"""


import torch 

from .sde import HeatDiffusion

def loss_fn(model, x, sde, eps=1e-5):

    """
    The loss function for training score-based generative models.
    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.
        sde: the forward sde
        eps: A tolerance value for numerical stability.
    """
    guided = (x.shape[1] == 2)
    if guided: # guided
        x_mri = x[:, 1, :,:].unsqueeze(1)
        x = x[:, 0, :, :].unsqueeze(1)
    
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)

    mean, std = sde.marginal_prob(x, random_t)  # for VESDE the mean is just x
    perturbed_x = mean + z * std[:, None, None, None]
    
    if guided:
        x_input = torch.cat([perturbed_x, x_mri], dim=1)
    else:
        x_input = perturbed_x

    score = model(x_input, random_t)

    if isinstance(sde, HeatDiffusion):
        """
        The loss function for training score-based generative models.
        Using the soft diffusion target from 
            Daras et al. (2022) [https://arxiv.org/pdf/2209.05442.pdf]
        """
        r_t = x - perturbed_x
        mean_model, _ = sde.marginal_prob(score - r_t, random_t)
        loss = torch.mean(torch.sum((std[:,None,None,None].pow(-1)*mean_model)**2, dim=(1,2,3)))

    else:
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    
    return loss
