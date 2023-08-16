from typing import Optional, Any, Dict, Tuple

import torch
import numpy as np

from torch import Tensor
from src.utils import SDE, VESDE, VPSDE
from src.third_party_models import OpenAiUNetModel

def Euler_Maruyama_sde_predictor(
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    time_step: Tensor,
    step_size: float,
    nloglik: Optional[callable] = None,
    datafitscale: Optional[float] = None,
    penalty: Optional[float] = None,
    aTweedy: bool = False,
    guidance_imgs: Optional[Tensor] = None,
    guidance_strength: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
    '''
    Implements the predictor step using Euler-Maruyama 
    (i.e., see Eq.30) in 
        1. @article{song2020score,
            title={Score-based generative modeling through stochastic differential equations},
            author={Song, Yang and Sohl-Dickstein, Jascha and Kingma,
                Diederik P and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
            journal={arXiv preprint arXiv:2011.13456},
            year={2020}
        }, available at https://arxiv.org/abs/2011.13456.
    If ``aTweedy'' is ``False'', it implements: ``Robust Compressed Sensing MRI with Deep Generative Priors''. 
        2. @inproceedings{NEURIPS2021_7d6044e9,
            author = {Jalal, Ajil and Arvinte, Marius and Daras, Giannis and Price, Eric and Dimakis, Alexandros G and Tamir, Jon},
            booktitle = {Advances in Neural Information Processing Systems},
            editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
            pages = {14938--14954},
            publisher = {Curran Associates, Inc.},
            title = {Robust Compressed Sensing MRI with Deep Generative Priors},
            url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/7d6044e95a16761171b130dcb476a43e-Paper.pdf},
            volume = {34},
            year = {2021}
        }. Be aware that the implementation departs from ``Jalal et al.'' as it does not use annealed Langevin MCMC.
    If ``aTweedy`` is ``True'', it implements the predictor method named ``Diffusion Posterior Sampling'', presented in 
        3. @article{chung2022diffusion,
            title={Diffusion posterior sampling for general noisy inverse problems},
            author={Chung, Hyungjin and Kim, Jeongsol and Mccann, Michael T and Klasky, Marc L and Ye, Jong Chul},
            journal={arXiv preprint arXiv:2209.14687},
            year={2022}
        }, available at https://arxiv.org/pdf/2209.14687.pdf.
    '''
    if nloglik is not None: assert (datafitscale is not None) and (penalty is not None)
    x.requires_grad_()
    
    if guidance_imgs is not None or guidance_strength is not None:
        zeros = torch.zeros_like(x)
        assert x.shape == guidance_imgs.shape, "x and guidance image need to have the same shape."
        s_c = score(torch.cat([x,guidance_imgs], dim=1), time_step)
        s_0 = score(torch.cat([x,zeros], dim=1), time_step)
        s = (1 + guidance_strength)*s_c - guidance_strength * s_0
        s = s.detach() if not aTweedy else s
    else:
        s = score(x, time_step).detach() if not aTweedy else score(x, time_step)
    
    if nloglik is not None:
        if aTweedy: xhat0 = _aTweedy(s=s, x=x, sde=sde, time_step=time_step)
        loss, scaling_factors = nloglik(x if not aTweedy else xhat0)
        # Multiply as the gradient of df(scale*x)/dx = scale*df(x)/dx
        nloglik_grad = torch.autograd.grad(outputs=loss.sum(), inputs=x)[0]
    drift, diffusion = sde.sde(x, time_step)
    _s = s
    # if ``penalty == 1/σ2'' and ``aTweedy'' is False : recovers Eq.4 in 1.
    
    if aTweedy and nloglik is not None: datafitscale =  loss[:, None, None].pow(-1)

    if nloglik is not None and not aTweedy: 
        _s = _s - penalty*nloglik_grad*datafitscale # minus for negative log-lik. 
        # for naive (Jalal) datafitscale should be i/N
    x_mean = x - (drift - diffusion[:, None, None, None].pow(2)*_s)*step_size
    noise = torch.sqrt(diffusion[:, None, None, None].pow(2)*step_size)*torch.randn_like(x)
    x = x_mean + noise

    if aTweedy:
        x = x - penalty*nloglik_grad*datafitscale

    return x.detach(), x_mean.detach(), scaling_factors.detach() if nloglik is not None else None

def Langevin_sde_corrector(
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    time_step: Tensor,
    nloglik: Optional[callable] = None,
    datafitscale: Optional[float] = None,
    penalty: Optional[float] = None,
    corrector_steps: int = 1,
    snr: float = 0.16,
    guidance_imgs: Optional[Tensor] = None,
    guidance_strength: Optional[float] = None,
    ) -> Tensor:

    ''' 
    Implements the corrector step using Langevin MCMC   

    This corrector step is not using Tweedie. 
    '''
    if nloglik is not None: assert (datafitscale is not None) and (penalty is not None)
    for _ in range(corrector_steps):
        x.requires_grad_()
        if guidance_imgs is not None or guidance_strength is not None:
            zeros = torch.zeros_like(x)
            assert x.shape == guidance_imgs.shape, "x and guidance image need to have the same shape."
            s_c = score(torch.cat([x,guidance_imgs], dim=1), time_step)
            s_0 = score(torch.cat([x,zeros], dim=1), time_step)
            s = (1 + guidance_strength)*s_c - guidance_strength * s_0
            s = s.detach()
        else:
            s = score(x, time_step).detach()
        if nloglik is not None: nloglik_grad = torch.autograd.grad(outputs=nloglik(x), inputs=x)[0]
        overall_grad = s - penalty*nloglik_grad*datafitscale if nloglik is not None else s
        overall_grad_norm = torch.norm(
                overall_grad.reshape(overall_grad.shape[0], -1), 
                dim=-1  ).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / overall_grad_norm)**2
        x = x + langevin_step_size * overall_grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
    return x.detach()


def _aTweedy(s: Tensor, x: Tensor, sde: SDE, time_step:Tensor) -> Tensor:

    update = x + s*sde.marginal_prob_std(time_step)[:, None, None, None].pow(2)
    div = sde.marginal_prob_mean(time_step)[:, None, None, None].pow(-1)
    return update*div

def chain_simple_init(
    time_steps: Tensor,
    sde: SDE, 
    filtbackproj: Tensor, 
    start_time_step: int, 
    im_shape: Tuple[int, int], 
    batch_size: int, 
    device: Any
    ) -> Tensor:

    t = torch.ones(batch_size, device=filtbackproj.device) * time_steps[start_time_step]
    print("START TIME STEP: ", t)
    mean, std = sde.marginal_prob(filtbackproj, t)
    return mean + torch.randn(batch_size, *im_shape, device=filtbackproj.device) * std[:, None, None, None]


def soft_diffusion_sde_predictor(
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    time_step: Tensor,
    step_size: float,
    nloglik: Optional[callable] = None, # not used
    datafitscale: Optional[float] = None, # not used
    penalty: Optional[float] = None, # not used
    aTweedy: bool = False # not used 
    ) -> Tuple[Tensor, Tensor]:
    '''
    Implements the naive soft diffusion sampler 
    (i.e., see Algorithm 1) in 
        1. @article{daras2022soft,
            title={Soft diffusion: Score matching for general corruptions},
            author={Daras, Giannis and Delbracio, Mauricio and Talebi, Hossein and Dimakis, Alexandros G and Milanfar, Peyman},
            journal={arXiv preprint arXiv:2209.05442},
            year={2022}
            }, available at https://arxiv.org/abs/2209.05442.
    '''
    
    with torch.no_grad():
        s = score(x, time_step)
    
    x0hat = s + x

    eta = torch.randn_like(x) 

    x_mean, std = sde.marginal_prob(x0hat, time_step - step_size)

    x = x_mean + std[:,None,None,None]*eta

    return x.detach(), x_mean.detach()

def soft_diffusion_momentum_sde_predictor(
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    time_step: Tensor,
    step_size: float,
    nloglik: Optional[callable] = None, # not used
    datafitscale: Optional[float] = None, # not used
    penalty: Optional[float] = None, # not used
    aTweedy: bool = False # not used 
    ) -> Tuple[Tensor, Tensor]:
    '''
    Implements the momentum soft diffusion sampler 
    (i.e., see Algorithm 2) in 
        1. @article{daras2022soft,
            title={Soft diffusion: Score matching for general corruptions},
            author={Daras, Giannis and Delbracio, Mauricio and Talebi, Hossein and Dimakis, Alexandros G and Milanfar, Peyman},
            journal={arXiv preprint arXiv:2209.05442},
            year={2022}
            }, available at https://arxiv.org/abs/2209.05442.
    '''
    
    with torch.no_grad():
        s = score(x, time_step)
    
    x0hat = s + x

    y_t, std_t = sde.marginal_prob(x0hat, time_step)
    y_tminus1, std_tminus1 = sde.marginal_prob(x0hat, time_step - step_size)
    eta_t = y_t - x 

    std_t_sq = std_t[:,None,None,None].pow(2)
    std_tminus1_sq = std_tminus1[:,None,None,None].pow(2)
    sigma = (std_tminus1_sq - std_t_sq)*std_t_sq.pow(-1)

    z_tminus1 = x - sigma*eta_t + torch.sqrt(std_t_sq -std_tminus1_sq)*torch.randn_like(x) 

    x = z_tminus1 + (y_tminus1 - y_t)

    return x.detach(), x0hat.detach()



def decomposed_diffusion_sampling_sde_predictor( 
    score: OpenAiUNetModel,
    sde: SDE,
    x: Tensor,
    time_step: Tensor,
    eta: float,
    step_size: float,
    nloglik: Optional[callable] = None, # this is the osem reco
    datafitscale: Optional[float] = None, # placeholder
    use_simplified_eqn: bool = False,
    guidance_imgs: Optional[Tensor] = None,
    guidance_strength: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:

    '''
    It implements ``Decomposed Diffusion Sampling'' for the SDE model 
        presented in 
            1. @article{chung2023fast,
                title={Fast Diffusion Sampler for Inverse Problems by Geometric Decomposition},
                author={Chung, Hyungjin and Lee, Suhyeon and Ye, Jong Chul},
                journal={arXiv preprint arXiv:2303.05754},
                year={2023}
            },
    available at https://arxiv.org/pdf/2303.05754.pdf. See Algorithm 4 in Appendix. 
    '''
    '''
    Implements the Tweedy denosing step proposed in ``Diffusion Posterior Sampling''.
    '''
    datafitscale = 1. # lace-holder

    #s = score(x, time_step).detach()
    if guidance_imgs is not None or guidance_strength is not None:
        zeros = torch.zeros_like(x)
        assert x.shape == guidance_imgs.shape, "x and guidance image need to have the same shape."
        s_c = score(torch.cat([x,guidance_imgs], dim=1), time_step)
        s_0 = score(torch.cat([x,zeros], dim=1), time_step)
        s = (1 + guidance_strength)*s_c - guidance_strength * s_0
        s = s.detach()
    else:
        if x.shape[0] > 50:
            list_indices = np.array_split(range(x.shape[0]), x.shape[0]//20)
            s = score(x[[list_indices[0]]], time_step[list_indices[0]]).detach()
            for i in list_indices[1:]:
                s_single = score(x[[i]], time_step[[i]]).detach()
                s = torch.cat([s, s_single], dim=0)
        else:
            s = score(x, time_step).detach()
    xhat0 = _aTweedy(s=s, x=x, sde=sde, time_step=time_step) # Tweedy denoising step
    # get xhat = OSEM starting with xhat0
    if nloglik is not None:
        xhat0, scaling_factors = nloglik(torch.clamp(xhat0, 0.))
    
    '''
    It implemets the predictor sampling strategy presented in
        2. @article{song2020denoising,
            title={Denoising diffusion implicit models},
            author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
            journal={arXiv preprint arXiv:2010.02502},
            year={2020}
        }, available at https://arxiv.org/pdf/2010.02502.pdf.
    '''
    x = _ddim_dds(sde=sde, s=s, xhat=xhat0, time_step=time_step, step_size=step_size, eta=eta, use_simplified_eqn=use_simplified_eqn)

    return x.detach(), xhat0, scaling_factors.detach() if nloglik is not None else None

def _ddim_dds(
    sde: SDE,
    s: Tensor,
    xhat: Tensor,
    time_step: Tensor,
    step_size: Tensor, 
    eta: float, 
    use_simplified_eqn: bool = False
    ) -> Tensor:
    
    if isinstance(sde, VESDE):
        """
        Following algorithm 4 from [https://arxiv.org/pdf/2303.05754.pdf]
        """

        std_t = sde.marginal_prob_std(t=time_step
            )[:, None, None, None]
        std_tminus1 = sde.marginal_prob_std(t=time_step-step_size
            )[:, None, None, None]
        tbeta = 1 - ( std_tminus1.pow(2) * std_t.pow(-2) ) if not use_simplified_eqn else torch.tensor(1.) 

        #noise_deterministic = - std_t*std_tminus1*torch.sqrt( 1 - tbeta.pow(2)*eta**2 ) * s
        noise_deterministic = - std_tminus1*std_tminus1*torch.sqrt( 1 - tbeta.pow(2)*eta**2 ) * s
        noise_stochastic = std_tminus1 * eta*tbeta*torch.randn_like(xhat)

    elif isinstance(sde, VPSDE):
        mean_tminus1 = sde.marginal_prob_mean(t=time_step-step_size
            )[:, None, None, None]
        mean_t = sde.marginal_prob_mean(t=time_step
            )[:, None, None, None]
        std_t = sde.marginal_prob_std(t=time_step
            )[:, None, None, None]
        tbeta = ((1 - mean_tminus1.pow(2)) / ( 1 - mean_t.pow(2) ) ).pow(.5) * (1 - mean_t.pow(2) * mean_tminus1.pow(-2) ).pow(.5) 
        #if tbeta.isnan():
        #    tbeta = torch.tensor(0)
        xhat = xhat*mean_tminus1

        # the DDIM sampling is given using a different parametrization of the score
        e = - std_t * s # s = - z/std_t

        noise_deterministic = torch.sqrt( 1 - mean_tminus1.pow(2) - tbeta.pow(2)*eta**2 )*e
        noise_stochastic = eta*tbeta*torch.randn_like(xhat)
    else:
        raise NotImplementedError

    return xhat + noise_deterministic + noise_stochastic


"""

#Chung et al. (2023) https://arxiv.org/pdf/2303.05754.pdf

import pyparallelproj.subsets as subsets
import pyparallelproj.algorithms as algorithms
import pyparallelproj.coincidences as coincidences

def dds_sampling(x, 
					score_model,
					marginal_prob_std_fn,
					diffusion_coeff_fn,
					time,
					step_size,
					acq_model, 
					measurements,
					loss,
					norm_factors,
					contamination,
					iteration_num, 
					attn_factors,
					config):


	if config.langevin:
		with torch.no_grad():
			grad = score_model(x, time)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = np.sqrt(np.prod(x.shape[1:]))
			langevin_step_size = 2 * (0.16 * noise_norm / grad_norm)**2
			x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

	eta = config.stochasticity
	x = x.requires_grad_()

	# DENOISE STEP
	with torch.no_grad():
		s = score_model(x, time)
		std = marginal_prob_std_fn(time)
		# DOUBLE CHECK THIS
		xhat0 = x + s * std[:, None, None, None]**2
		xhat0 = torch.clamp(xhat0, min=1e-3) * norm_factors

	# SET THE SUBSETTER
	subsetter = subsets.SingoramViewSubsetter(acq_model._coincidence_descriptor, config.num_subsets)
	acq_model.subsetter = subsetter

	x_mean = []
	for sample in range(x.shape[0]):

		# UPDATE THE MULTIPLICATIVE CORRECTIONS
		a_f = xp.asarray(attn_factors[sample,0,:])
		acq_model.multiplicative_corrections = a_f/30.

		c = xp.asarray(contamination[sample,0,:])
		m = xp.asarray(measurements[sample,0,:])
		reconstructor = algorithms.OSEM(data = m, 
				  contamination = c*xp.ones_like(m), 
				  data_operator = acq_model,
				  verbose=False)
		reconstructor.setup(xp.asarray(xhat0[sample,0,:,:].unsqueeze(-1)))
		reconstructor.run(config.num_epochs, evaluate_cost=False)
		x_mean.append(torch.from_dlpack(reconstructor.x))

	x_mean = torch.stack(x_mean).squeeze().unsqueeze(1)/norm_factors

	std_t = marginal_prob_std_fn(time)[:,None,None,None]
	std_tminus1 = marginal_prob_std_fn(time - step_size)[:,None,None,None]

	if config.beta_unit:
		beta = 1.
	else:
		beta = 1 - std_tminus1**2/std_t**2

	noise_deterministic = -std_tminus1*std_t*torch.sqrt(1- beta**2*eta**2)*s
	noise_stochastic = std_tminus1*eta*beta* torch.randn_like(x)

	x = x_mean + noise_deterministic + noise_stochastic

	fwd_proj = LPDForwardFunction2D.apply(x_mean.requires_grad_(), acq_model, attn_factors/30.)
	fwd_proj = fwd_proj + contamination
	loss_vals = loss(fwd_proj, measurements)
	likelihood_grad = torch.autograd.grad(outputs=loss_vals, inputs=x_mean.requires_grad_())[0]

	return x.detach(), x_mean.detach(), likelihood_grad.detach(), loss_vals.item()

"""