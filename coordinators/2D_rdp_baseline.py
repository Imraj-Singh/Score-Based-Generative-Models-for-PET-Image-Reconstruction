import matplotlib.pyplot as plt
import torch, sys, os
import pyparallelproj.coincidences as coincidences
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
import cupy as xp
import cupyx.scipy.ndimage as ndi
from tqdm import tqdm
sys.path.append(os.path.dirname(os.getcwd()))
from src import BrainWebOSEM, LPDForwardFunction2D, LPDAdjointFunction2D


detector_efficiency = 1./30
coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
    num_rings=1,
    sinogram_spatial_axis_order=coincidences.SinogramSpatialAxisOrder['RVP'],xp=xp)
acq_model = petprojectors.PETJosephProjector(coincidence_descriptor,
                                            (128, 128, 1), (-127.0, -127.0, 0.0),
                                            (2., 2., 2.))
res_model = resolution_models.GaussianImageBasedResolutionModel(
            (128, 128, 1), tuple(4.5 / (2.35 * x) for x in (2., 2., 2.)), xp, ndi)
acq_model.image_based_resolution_model = res_model



def pnll_gradient(x, measurements, acq_model, attn_factor, contamination_factor):
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
    # from A Concave Prior Penalizing Relative Differences
    # for Maximum-a-Posteriori Reconstruction
    # in Emission Tomography Eq. 15
    first = sens_img/x
    x = x.repeat(1,9,1,1)
    second = (16*(x_neighbours**2))/torch.clamp(((x + x_neighbours + 2 * torch.abs(x-x_neighbours))**3),0)
    return 1/(first - beta*second.sum(dim=1, keepdim=True))

def rdp_gradient(x, x_neighbours):
    x = x.repeat(1,9,1,1)
    numerator = (x - x_neighbours)*(2 * torch.abs(x-x_neighbours) + x + 3 * x_neighbours)
    denominator = (x + x_neighbours + 2 * torch.abs(x - x_neighbours))**2
    return - (numerator/denominator).sum(dim=1, keepdim=True)

def obj_value(x, x_neighbours, measurements, acq_model, attn_factors, contamination_factor, beta):
    y_pred = LPDForwardFunction2D.apply(x, acq_model, attn_factors) + contamination_factor
    kl = - (measurements*torch.log(measurements/y_pred+1e-9) + (y_pred-measurements)).sum(-1)
    x = x.repeat(1,9,1,1)
    numerator = (x - x_neighbours)**2
    denominator = (x + x_neighbours + 2 * torch.abs(x - x_neighbours))
    rdp = - beta*(numerator/denominator).sum(dim=1, keepdim=True).sum(-1).sum(-1)
    return (kl + rdp).mean()

def compute_kl_div(recons, measurements, acq_model, attn_factors, contamination_factor):
	kldiv_r = []
	for r in range(len(recons)):
		y_pred = LPDForwardFunction2D.apply(recons[[r]], acq_model, attn_factors[[r]]) + contamination_factor[[r],..., None]
		kl = (measurements[[r]]*torch.log(measurements[[r]]/y_pred+1e-9)+ (y_pred-measurements[[r]])).sum()
		if kl.isnan():
			print("KL is nan")
		kldiv_r.append(kl)
	return  torch.asarray(kldiv_r).cpu()


parts = ["test", "test_tumour"]
noises = [10]
for noise in noises:
    for part in parts:
        dataset = BrainWebOSEM(part=part,
                noise_level=noise, 
                base_path="path_tof/src/brainweb_2d/")
        subset = list(range(2, len(dataset), 4))
        dataset = torch.utils.data.Subset(dataset, subset)
        betas = [0.11,0.1,0.09,0.075,0.05,0.025,0.01,0.001]
        for beta in betas:
            save_recon = []
            save_ref = []
            save_kldivs = []
            save_lesion_rois = []
            save_background_rois = []
            print(f"beta: {beta}")
            idx = 0
            for batch in dataset:
                idx += 1
                # [0] reference, [1] scale_factor, [2] osem, [3] norm, [4] measurements,
                # [5] contamination_factor, [6] attn_factors
                gt = batch[0].to("cuda:0").unsqueeze(1)[...]
                osem = batch[2].to("cuda:0").unsqueeze(1)[...]
                measurements = batch[4].to("cuda:0").unsqueeze(1)[...]
                contamination_factor = batch[5].to("cuda:0")[:,None,None]
                attn_factors = batch[6].to("cuda:0").unsqueeze(1)[...]*detector_efficiency
                sens_img = LPDAdjointFunction2D.apply(torch.ones_like(measurements), acq_model, attn_factors).detach()
                x_old = osem.clone().detach()
                grad_norm = []
                objective_values = []
                x_neighbours = rdp_rolled_components(x_old)
                prev_obj =  obj_value(x_old, x_neighbours, measurements, acq_model, attn_factors, contamination_factor, beta)
                for i in tqdm(range(1000)):
                    x_neighbours = rdp_rolled_components(x_old)
                    preconditioner = get_preconditioner(x_old, x_neighbours, sens_img, beta).detach()
                    gradient = (pnll_gradient(x_old, measurements, acq_model, attn_factors, contamination_factor) 
                        + beta * rdp_gradient(x_old, x_neighbours)).detach()
                    
                    x_new = torch.clamp(x_old.detach() + \
                        preconditioner * gradient, 0)
                    x_old = x_new
                    new_obj = obj_value(x_new, x_neighbours, measurements, acq_model, attn_factors, contamination_factor, beta)
                    if (new_obj - prev_obj)**2 < 1e-6:
                        break
                    grad_norm.append(torch.norm(gradient).item())
                    objective_values.append(new_obj.item())
                    prev_obj = new_obj.clone()
                    kldiv_r = compute_kl_div(recons = x_new,
                            measurements=measurements,
                            acq_model=acq_model, 
                            attn_factors=attn_factors, 
                            contamination_factor=contamination_factor)
                
                lesion_roi = batch[-1].to("cuda:0")
                background_roi = batch[-2].to("cuda:0")
                
                save_recon.append(x_new.squeeze().cpu())
                save_ref.append(gt.squeeze().cpu())
                save_kldivs.append(kldiv_r)
                save_lesion_rois.append(lesion_roi.squeeze().cpu())
                save_background_rois.append(background_roi.squeeze().cpu())

                print("iterations: ", i)
                fig, ax = plt.subplots(1,5, figsize=(25,5))
                fig.colorbar(ax[0].imshow(gt[0,0].cpu().numpy()))
                ax[0].set_title("Ground truth")
                ax[0].axis("off")
                fig.colorbar(ax[1].imshow(sens_img[0,0].cpu().numpy()))
                ax[1].set_title("Sensitivity image")
                ax[1].axis("off")
                fig.colorbar(ax[2].imshow(osem[0,0].cpu().numpy()))
                ax[2].set_title("OSEM")
                ax[2].axis("off")
                fig.colorbar(ax[3].imshow(x_new[0,0].cpu().numpy()))
                ax[3].set_title(f"Penalised MAP beta: {beta}")
                ax[3].axis("off")
                ax[4].plot(objective_values)
                ax[4].set_title("Objective value")
                plt.savefig(f"path_to/coordinators/RDP/{part}_{noise}/rdp_baseline_image_{idx}_beta_{beta}.png", dpi=300, bbox_inches="tight")
                plt.close()
                #plt.show()
            
            results = {"images": torch.stack(save_recon).cpu(),
                "ref": torch.stack(save_ref).cpu(),
                "kldiv": torch.stack(save_kldivs).cpu(),
                "lesion_rois": torch.stack(save_lesion_rois).cpu(), 
                "background_rois": torch.stack(save_background_rois).cpu(),
                "beta": beta}

            name = f"rdp_baseline_beta_{beta}"
            torch.save(results, f"path_to/coordinators/RDP/{part}_{noise}/{name}.pt")