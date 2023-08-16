import torch 

import pyparallelproj.algorithms as algorithms
import pyparallelproj.subsets as subsets

import cupy as xp
import time
from ..sirf.herman_meyer import herman_meyer_order

from ..brainweb_2d import LPDForwardFunction2D, LPDAdjointFunction2D


def kl_div(x, acq_model, attn_factors, contamination, measurements, scale_factor):
    fwd_proj = LPDForwardFunction2D.apply(torch.clamp(x, 0) * scale_factor, acq_model, attn_factors/30.)
    fwd_proj = fwd_proj + contamination
    kl = torch.sum(measurements * torch.log(measurements/fwd_proj + 1e-6) - measurements + fwd_proj, axis=[-1])
    return kl, scale_factor

def poisson_nll(x, acq_model, attn_factors, contamination, measurements, scale_factor, loss):
    """ print(scale_factor.squeeze().tolist())
    x = torch.clamp(x, 0)
    sens_img = LPDAdjointFunction2D.apply(torch.ones_like(measurements), acq_model, attn_factors).detach()
    scale_factor = (measurements-torch.ones_like(measurements)*contamination).sum(dim=(1,2))/(x*sens_img).sum(dim=(1,2,3))
    scale_factor = scale_factor.detach()[:,None,None,None]
    scale_factor = torch.clamp(scale_factor, 1e-6, 1000)
    print(scale_factor.squeeze().tolist()) """
    fwd_proj = LPDForwardFunction2D.apply(torch.clamp(x, 0) * scale_factor, acq_model, attn_factors/30.)
    fwd_proj = fwd_proj + contamination[:,None]
    #grad = sens_img - LPDAdjointFunction2D.apply((measurements) / (fwd_proj + 1e-9), acq_model, attn_factors/30.)
    loss_vals = loss(fwd_proj, measurements)
    #grad = torch.zeros_like(x)
    return loss_vals, scale_factor
	
def osem_nll(x, scale_factor, osem):
    loss_vals = (x * scale_factor - osem)**2
    return torch.sum(loss_vals, axis=[-1, -2]), scale_factor
	
# TODO: Instead of OSEM, do gradient descent on min datafit(Ax,y) + beta*||x-x0hat||^2
def get_osem(x, acq_model, attn_factors, contamination, measurements, scale_factor, num_subsets, num_epochs):
    # SET THE SUBSETTER
    subsetter = subsets.SingoramViewSubsetter(acq_model._coincidence_descriptor, num_subsets)
    acq_model.subsetter = subsetter
    x_mean = []
    for sample in range(x.shape[0]):
        # UPDATE THE MULTIPLICATIVE CORRECTIONS
        a_f = xp.asarray(attn_factors[sample,0,:])
        acq_model.multiplicative_corrections = a_f/30.
        c = xp.asarray(contamination[sample,0])
        m = xp.asarray(measurements[sample,0,:])
        reconstructor = algorithms.OSEM(data = m, 
            contamination = c*xp.ones_like(m), 
            data_operator = acq_model,
            verbose=False)
        reconstructor.setup(xp.asarray(x[sample,0,:,:].unsqueeze(-1)*scale_factor[sample, ...]))
        reconstructor.run(num_epochs, evaluate_cost=False)
        x_mean.append(torch.from_dlpack(reconstructor.x))
    x_mean = torch.stack(x_mean).squeeze().unsqueeze(1)/scale_factor
    return x_mean, scale_factor

def pll_gradient(x, measurements, acq_model, attn_factor, contamination_factor):
    tmp = measurements / (LPDForwardFunction2D.apply(x, acq_model, attn_factor) + contamination_factor) - torch.ones_like(measurements)
    return LPDAdjointFunction2D.apply(tmp, acq_model, attn_factor)

def rdp_rolled_components(x):
    rows = [1,1,1,0,0,0,-1,-1,-1]
    columns = [1,0,-1,1,0,-1,1,0,-1]
    x_neighbours = x.clone().repeat(1,9,1,1)
    for i in range(9):
        x_neighbours[:,[i]] = torch.roll(x, shifts=(rows[i], columns[i]), dims=(-2, -1))
    return x_neighbours

def get_preconditioner(x, x_neighbours, sens_img, beta):
    first = torch.clamp(sens_img,1e-9)/(x+1e-9)
    x = x.repeat(1,9,1,1)
    second = (16*(x_neighbours**2))/torch.clamp((x + x_neighbours + 2 * torch.abs(x-x_neighbours))**3,1e-9)
    return 1/(first - beta*second.sum(dim=1, keepdim=True))

def rdp_gradient(x, x_neighbours):
    x = x.repeat(1,9,1,1)
    numerator = (x - x_neighbours)*(2 * torch.abs(x-x_neighbours) + x + 3 * x_neighbours)
    denominator = torch.clamp((x + x_neighbours + 2 * torch.abs(x - x_neighbours))**2,1e-9)
    return - (numerator/denominator).sum(dim=1, keepdim=True)

def get_map(x, acq_model, attn_factors, contamination, measurements, scale_factor, num_subsets, num_epochs, beta):
    # from A Concave Prior Penalizing Relative Differences
    # for Maximum-a-Posteriori Reconstruction
    # in Emission Tomography Eq. 15
    sens_img = LPDAdjointFunction2D.apply(torch.ones_like(measurements), acq_model, attn_factors/30.).detach()
    x_old = torch.clamp(scale_factor*x.clone().detach(),0)
    for _ in range(num_epochs):
        x_neighbours = rdp_rolled_components(x_old)
        preconditioner = get_preconditioner(x_old, x_neighbours, sens_img, beta).detach()
        gradient_1 = pll_gradient(x_old, measurements, acq_model, attn_factors/30., contamination).detach()
        gradient_2 = beta * rdp_gradient(x_old, x_neighbours).detach()
        x_new = torch.clamp(x_old.detach() + \
            preconditioner * (gradient_1 + gradient_2), 0)
        x_old = x_new
    return x_new/scale_factor, scale_factor

def get_anchor(x, acq_model, attn_factors, contamination, measurements, scale_factor, num_subsets, num_epochs, beta):
    x_anchor = scale_factor*x.clone().detach()
    sens_img = LPDAdjointFunction2D.apply(torch.ones_like(measurements), acq_model, attn_factors/30.).detach()
    x_old = scale_factor*x.clone().detach()
    for _ in range(num_epochs):
        preconditioner = (x_old + 1e-9)/torch.clamp(sens_img,1e-9)
        gradient_1 = pll_gradient(x_old, measurements, acq_model, attn_factors/30., contamination).detach()
        gradient_2 = beta * (x_anchor-x_old).detach()
        x_new = torch.clamp(x_old.detach() + \
            preconditioner * (gradient_1 + gradient_2), 1e-9)
        
        x_old = x_new
    
    return x_new/scale_factor, scale_factor