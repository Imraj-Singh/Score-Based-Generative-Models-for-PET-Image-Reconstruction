import hydra
import torch
import functools
import numpy as np 
import yaml
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from src import (BrainWebOSEM, get_standard_score, get_standard_sde,
		LPDForwardFunction2D, get_standard_sampler, osem_nll,
		get_osem, get_map, get_anchor, kl_div)
from omegaconf import DictConfig, OmegaConf
import torchvision
import matplotlib.pyplot as plt 
from src import PSNR, SSIM

import cupy as xp

# not used in this script
#detector_efficiency = 1./30


def get_acq_model():
	import pyparallelproj.coincidences as coincidences
	import pyparallelproj.petprojectors as petprojectors
	import pyparallelproj.resolution_models as resolution_models
	import cupyx.scipy.ndimage as ndi

	"""
	create forward operator
	"""
	coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
		num_rings=1,
		sinogram_spatial_axis_order=coincidences.SinogramSpatialAxisOrder['RVP'],xp=xp)
	acq_model = petprojectors.PETJosephProjector(coincidence_descriptor,
		(128, 128, 1), (-127.0, -127.0, 0.0), (2., 2., 2.))
	res_model = resolution_models.GaussianImageBasedResolutionModel(
		(128, 128, 1), tuple(4.5 / (2.35 * x) for x in (2., 2., 2.)), xp, ndi)
	acq_model.image_based_resolution_model = res_model
	return acq_model


def estimate_scale_factor(osem, measurements, contamination, normalisation_type):
	scale_factors = []		
	for i in range(osem.shape[0]):
		if normalisation_type == "data_scale":
			emission_volume = torch.where(osem[i] > 0.01*osem[i].max(), 1, 0).sum() * 8
			scale_factor = (measurements[i] - contamination[i]).sum()/emission_volume
			scale_factors.append(scale_factor)
		elif normalisation_type == "image_scale":
			emission_volume = torch.where(osem[i] > 0.01*osem[i].max(), 1, 0).sum()
			scale_factor = osem[i].sum()/emission_volume
			scale_factors.append(scale_factor)		
		else:
			raise NotImplementedError
	return torch.tensor(scale_factors)

def compute_kl_div(recons, measurements, acq_model, attn_factors, contamination_factor):
	kldiv_r = []
	for r in range(len(recons)):
		y_pred = LPDForwardFunction2D.apply(recons[[r]], acq_model, attn_factors[[r]]) + contamination_factor[[r]]
		kl = (measurements[[r]]*torch.log(measurements[[r]]/y_pred+1e-9)+ (y_pred-measurements[[r]])).sum()
		if kl.isnan():
			print(torch.log(measurements[[r]]/y_pred[[r]]+1e-9).sum())
			print("KL is nan")
		kldiv_r.append(kl)
	return  torch.asarray(kldiv_r).cpu()

@hydra.main(config_path='../configs', config_name='test_reconstruction')
def reconstruction(config : DictConfig) -> None:
	print(OmegaConf.to_yaml(config))
	
	###### SET SEED ######
	if config.seed is not None:
		torch.manual_seed(config.seed)
		np.random.seed(config.seed)

	###### GET SCORE MODEL ######
	# open the yaml config file
	with open(os.path.join(config.score_based_model.path, "report.yaml"), "r") as stream:
		ml_collection = yaml.load(stream, Loader=yaml.UnsafeLoader)
	guided = False if ml_collection.guided_p_uncond is None else True
	# get the sde
	sde = get_standard_sde(ml_collection)
	# get the score model
	score_model = get_standard_score(ml_collection, sde, 
		use_ema = config.score_based_model.ema, 
		load_path = config.score_based_model.path)
	score_model.eval()
	score_model.to(config.device)
	
	###### GET ACQUISITION MODEL AND DATA ######
	# get the acquisition model
	acq_model = get_acq_model()
	# get the data
	dataset = BrainWebOSEM(part=config.dataset.part,
			noise_level=config.dataset.poisson_scale, 
			base_path=config.dataset.base_path,
			guided=guided)
	test_loader = torch.utils.data.DataLoader(dataset, 
		batch_size=8, shuffle=False)

	config.sampling.batch_size = 8

	###### SOLVING REVERSE SDE ######
	img_shape = (config.dataset.img_z_dim, 
	    config.dataset.img_xy_dim, config.dataset.img_xy_dim)

	for idx, batch in enumerate(test_loader):
		# [0] reference, [1] scale_factor, [2] osem, [3] norm, [4] measurements, [5] contamination_factor, [6] attn_factors
		if guided:
			gt = batch[0][:, [0], ...]
			guided_img = batch[0][:, [1], ...].to(config.device)
		else:
			gt = batch[0][:, [0], ...]
			guided_img = None 

		print("Normalisation type: ", ml_collection.normalisation)
		attn_factors=batch[6][:,[0],...].to(config.device)
		contamination_factor=batch[5][:,[0],None].to(config.device)
		measurements=batch[4][:,[0],...].to(config.device)
		osem=batch[2][:,[0],...]
		gt=batch[0][:, [0], ...]
		
		# estimate scaling factors from measurements
		scale_factor = estimate_scale_factor(osem=osem, 
			measurements=measurements, contamination=contamination_factor,
			normalisation_type=ml_collection.normalisation)[:, None, None, None].to(config.device)

		if config.sampling.use_osem_nll:
			nll_partial = functools.partial(osem_nll, 
				scale_factor=scale_factor,
				osem=osem.to(config.device))
		elif config.sampling.name == "dds":
			if config.sampling.dds_proj.name == "osem":
				nll_partial = functools.partial(get_osem,
					acq_model=acq_model,
					attn_factors=attn_factors,
					contamination=contamination_factor,
					measurements=measurements,
					scale_factor=scale_factor,
					num_subsets=config.sampling.dds_proj.num_subsets,
					num_epochs=config.sampling.dds_proj.num_epochs)
			elif config.sampling.dds_proj.name == "map":
				nll_partial = functools.partial(get_map,
					acq_model=acq_model, 
					attn_factors=attn_factors, 
					contamination=contamination_factor, 
					measurements=measurements, 
					scale_factor=scale_factor, 
					num_subsets=config.sampling.dds_proj.num_subsets, 
					num_epochs=config.sampling.dds_proj.num_epochs,
					beta = config.sampling.dds_proj.beta)
			elif config.sampling.dds_proj.name == "anchor":
				nll_partial = functools.partial(get_anchor,
					acq_model=acq_model, 
					attn_factors=attn_factors, 
					contamination=contamination_factor, 
					measurements=measurements, 
					scale_factor=scale_factor, 
					num_subsets=config.sampling.dds_proj.num_subsets, 
					num_epochs=config.sampling.dds_proj.num_epochs,
					beta = config.sampling.dds_proj.beta)
			else:
				raise NotImplementedError
		else:
			nll_partial = functools.partial(kl_div, 
				acq_model=acq_model, 
				attn_factors=attn_factors, 
				contamination=contamination_factor, 
				measurements=measurements,
				scale_factor=scale_factor)


		logg_kwargs = {'log_dir': "./tb", 'num_img_in_log': 10,
			'sample_num':idx, 'ground_truth': gt, 'osem': osem}
		
		sampler = get_standard_sampler(
			config=config,
			score=score_model,
			sde=sde,
			nll=nll_partial, 
			im_shape=img_shape,
			guidance_imgs=guided_img if guided else None,
			device=config.device)
		
		recon, writer = sampler.sample(logg_kwargs=logg_kwargs)
		recon = torch.clamp(recon, min=0)
		recon = recon*scale_factor.to(config.device)
		
		fig, axes = plt.subplots(1,recon.shape[0])

		for idx, ax in enumerate(axes.ravel()):
			ax.imshow(recon[idx,0,:,:].cpu().numpy())

		plt.show()

		fig, (ax1, ax2, ax3) = plt.subplots(1,3)
		psnr = PSNR(recon[4].squeeze().cpu().numpy(), gt[4].squeeze().cpu().numpy())
		ax1.imshow(gt[4,0,:,:].cpu().numpy())
		ax2.imshow(recon[4,0,:,:].cpu().numpy())
		ax2.set_title(str(np.round(float(psnr), 4)))
		psnr = PSNR(osem[4].squeeze().cpu().numpy(), gt[4].squeeze().cpu().numpy())
		ax3.imshow(osem[4,0,:,:].cpu().numpy())
		ax3.set_title(str(np.round(float(psnr), 4)))

		plt.show() 

		kldiv_r = kl_div(x = recon,
			acq_model=acq_model, 
			attn_factors=attn_factors, 
			contamination=contamination_factor, 
			measurements=measurements,
			scale_factor=1.)[0]
		
		writer.add_image(
			'final_reco', torchvision.utils.make_grid(recon,
			normalize=True, scale_each=True), global_step=0)
		for i in range(config.sampling.batch_size):
			writer.add_scalar('PSNR_per_validation_img', PSNR(recon[i].squeeze().cpu().numpy(), gt[i].squeeze().cpu().numpy()), global_step=i)
			writer.add_scalar('SSIM_per_validation_img', SSIM(recon[i].squeeze().cpu().numpy(), gt[i].squeeze().cpu().numpy()), global_step=i)
			writer.add_scalar('kldiv_r', kldiv_r[i].mean(), global_step=i)
		writer.close()

if __name__ == '__main__':
    reconstruction()