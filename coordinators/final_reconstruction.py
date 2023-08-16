import hydra
import torch
import functools
import numpy as np 
import yaml
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from src import (BrainWebOSEM, get_standard_score, get_standard_sde, 
		 get_standard_sampler, osem_nll, get_osem, get_map, get_anchor, kl_div)
from omegaconf import DictConfig, OmegaConf
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import matplotlib.pyplot as plt 

import cupy as xp

detector_efficiency = 1./30

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

@hydra.main(config_path='../configs', config_name='final_reconstruction')
def reconstruction(config : DictConfig) -> None:
	print(OmegaConf.to_yaml(config))

	# Generate a unique filename, the file directories specify the SDE and data
	name = ""
	results = {}
	if config.sampling.use_osem_nll:
		name = "OSEMNLL_"
	if config.sampling.name == "dps" or config.sampling.name == "naive":
		name = name + "penalty_" + str(config.sampling.penalty) + "_"
		results["penalty"] = config.sampling.penalty
	if config.sampling.name == "dds":
		if config.sampling.dds_proj.name == "osem":
			name = name + "osem_num_epochs_" + str(config.sampling.dds_proj.num_epochs) + "_"
			results["num_epochs"] = config.sampling.dds_proj.num_epochs
		if config.sampling.dds_proj.name == "anchor":
			name = name + "anchor_num_epochs_" + str(config.sampling.dds_proj.num_epochs) + "_"
			results["num_epochs"] = config.sampling.dds_proj.num_epochs
			name = name + "_beta_" + str(config.sampling.dds_proj.beta) + "_"
			results["beta"] = config.sampling.dds_proj.beta
	if "guided" in config.score_based_model.name:
		name = name  + "_gstrength_" + str(config.sampling.guidance_strength) + "_"
		results["gstrength"] = config.sampling.guidance_strength

	timestr = time.strftime("%Y%m%d_%H%M%S_")
	dump_name = config.dump_path + "/"+ timestr + name + ".tmp"
	with open(dump_name, "xt") as f:
		f.write(os.getcwd())
		f.close()

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
	subset = list(range(2, len(dataset), 4))
	dataset = torch.utils.data.Subset(dataset, subset)
	test_loader = torch.utils.data.DataLoader(dataset, 
		batch_size=1, shuffle=False)
	# as there are 10 realisations then batch = 10
	config.sampling.batch_size = 10

	###### SOLVING REVERSE SDE ######
	img_shape = (config.dataset.img_z_dim, 
	    config.dataset.img_xy_dim, config.dataset.img_xy_dim)

	save_recon = []
	save_ref = []
	save_kldivs = []
	if "tumour" in config.dataset.part:
		save_lesion_rois = []
		save_background_rois = []
	if guided:
		save_guided = []
	print("Normalisation type: ", ml_collection.normalisation)
	print("Length of test loader: ", len(test_loader))
	for idx, batch in enumerate(test_loader):
		# [0] reference, [1] scale_factor, [2] osem, [3] norm, [4] measurements,
		# [5] contamination_factor, [6] attn_factors
		# FIRST STEP
		# swap axis so realisation are a batch
		osem = torch.swapaxes(batch[2], 0, 1).to(config.device)
		measurements = torch.swapaxes(batch[4], 0, 1).to(config.device)
		contamination_factor = torch.swapaxes(batch[5], 0, 1)[:,[0],None].to(config.device)
		attn_factors = torch.swapaxes(batch[6], 0, 1).to(config.device)

		gt = batch[0][:, [0], ...]
		if guided:
			guided_img = batch[0][:, [1], ...].repeat(config.sampling.batch_size, 1, 1, 1).to(config.device)

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

		logg_kwargs = {'log_dir': "./tb", 
			'num_img_in_log': config.sampling.batch_size, 'sample_num':idx,
			'ground_truth': gt, 'osem': None}
		
		sampler = get_standard_sampler(
			config=config,
			score=score_model,
			sde=sde,
			nll=nll_partial, 
			im_shape=img_shape,
			guidance_imgs=guided_img if guided else None,
			device=config.device)
		
		recon, _ = sampler.sample(logg_kwargs=logg_kwargs, logging=False)
		recon = torch.clamp(recon, min=0)
		recon = recon*scale_factor

		kldiv_r = kl_div(x = recon,
			acq_model=acq_model, 
			attn_factors=attn_factors, 
			contamination=contamination_factor, 
			measurements=measurements,
			scale_factor=1.)[0].squeeze()
		
		if "tumour" in config.dataset.part:
			lesion_roi = batch[-1].to(config.device)
			background_roi = batch[-2].to(config.device)
			save_lesion_rois.append(lesion_roi.squeeze().cpu())
			save_background_rois.append(background_roi.squeeze().cpu())
		
		save_recon.append(recon.squeeze().cpu())
		save_ref.append(gt.squeeze().cpu())
		save_kldivs.append(kldiv_r)
		if guided:
			save_guided.append(guided_img.squeeze()[0].cpu())
	
	if "tumour" in config.dataset.part:
		results["images"] = torch.stack(save_recon).cpu()
		results["ref"] = torch.stack(save_ref).cpu()
		results["kldiv"] = torch.stack(save_kldivs).cpu()
		results["lesion_rois"] = torch.stack(save_lesion_rois).cpu()
		results["background_rois"] = torch.stack(save_background_rois).cpu()
	else:
		results["images"] = torch.stack(save_recon).cpu()
		results["ref"] = torch.stack(save_ref).cpu()
		results["kldiv"] = torch.stack(save_kldivs).cpu()
		
	torch.save(results, name+".pt")
	os.remove(dump_name)
		 
if __name__ == '__main__':
    reconstruction()