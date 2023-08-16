import hydra, torch, yaml, sys, os, functools, time
import numpy as np 
sys.path.append(os.path.dirname(os.getcwd()))
from src import (get_standard_sampler, get_standard_score, get_standard_sde, SIRF3DProjection, SIRF3DDataset)
from omegaconf import DictConfig, OmegaConf
import sirf.STIR as pet
pet.set_verbosity(0)
pet.AcquisitionData.set_storage_scheme("memory")
pet.MessageRedirector(info=None, warn=None, errr=None)
import gc
# python 3D_reconstruction.py --multirun dataset=thorax3Dmedium sampling.lambd=0.1

@hydra.main(config_path='../configs', config_name='3D_reconstruction')
def reconstruction(config : DictConfig) -> None:
	print(OmegaConf.to_yaml(config))
	timestr = time.strftime("%Y%m%d_%H%M%S_")
	dump_name = config.dump_path + "/"+ timestr  + ".tmp"
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
	if guided: raise NotImplementedError
	# get the sde
	sde = get_standard_sde(ml_collection)
	# get the score model
	score_model = get_standard_score(ml_collection, sde, 
		use_ema = config.score_based_model.ema, 
		load_path = config.score_based_model.path)
	score_model.eval()
	score_model.to(config.device)
	
	###### SOLVING REVERSE SDE ######
	num_subsets = config.sampling.num_subsets
	num_iterations = config.sampling.num_iterations
	count_level = config.dataset.count_level

	# GET THE DATA
	# measurements_subsets_sirf, acquisition_models_sirf, sensitivity_image, fov, image_sirf, osem, measurements_subsets


	dataset = SIRF3DDataset(config.dataset.base_path, 
			config.dataset.name, 
			config.dataset.tracer, 
			count_level,
			config.dataset.realisation,
			num_subsets)

	image = dataset.osem
	config.sampling.batch_size = len(image)
	# estimate scaling factors from osem
	scale_factors = []
	for i in range(config.sampling.batch_size):
		emission_volume = torch.where(image[i] > 0.01*image[i].max(), 1, 0).sum()
		scale_factor = image[i].sum()/emission_volume
		# Less than 100 voxels in emmision volume then set scale factor to 0
		if emission_volume < 100:
			scale_factor = 0
		scale_factors.append(scale_factor)
	scale_factors = torch.tensor(scale_factors).to(config.device)
	# Remove outliers from scale factors
	scale_factors[scale_factors < scale_factors.mean()*0.05] = 0
	scale_factors = scale_factors.unsqueeze(1).unsqueeze(2).unsqueeze(3)

	
	if config.sampling.name == "dds_3D":
		projection = SIRF3DProjection(image_sirf = dataset.image_sirf.clone(), 
				objectives_sirf = dataset.objectives_sirf,
				sensitivity_image = dataset.sensitivity_image.clone(),
				fov = dataset.fov.clone(), 
				num_subsets = num_subsets, 
				num_iterations = num_iterations)
		if config.sampling.lambd != None and config.sampling.beta == None:
			print(f"Using DDS with anchor with lambda {config.sampling.lambd}")
			projection.set_lambd(config.sampling.lambd)
			nll_partial = functools.partial(projection.get_anchor,
				scale_factor=scale_factors)
		elif config.sampling.lambd != None and config.sampling.beta != None:
			print(f"Using DDS with anchor and rdpz with lambda {config.sampling.lambd} and beta {config.sampling.beta}")
			projection.set_lambd(config.sampling.lambd)
			projection.set_beta(config.sampling.beta)
			nll_partial = functools.partial(projection.get_anchor_rdpz,
				scale_factor=scale_factors)
		else: raise NotImplementedError("3D only DDS with anchor or anchor+rdpz")
	else: raise NotImplementedError("3D only tested with DDS")


	logg_kwargs = {'log_dir': "./tb", 'num_img_in_log': None,
		'sample_num': None, 'ground_truth': None, 'osem': None}
	sampler = get_standard_sampler(
		config=config,
		score=score_model,
		sde=sde,
		nll=nll_partial, 
		im_shape=(1,128,128),
		guidance_imgs= None,
		device=config.device)
	t0 = time.time()
	recon, _ = sampler.sample(logg_kwargs=logg_kwargs, logging=False)
	t1 = time.time()
	recon = torch.clamp(recon, min=0)
	recon = recon*scale_factors.to(config.device)
	torch.save(recon.detach().cpu().squeeze().swapaxes(1,2).flip(1,2), "volume.pt")
	recon_dict = {"pll_values": projection.pll_values,
					"objective_values": projection.objective_values,
					"pll_values_last": projection.pll_values_last,
					"objective_values_last": projection.objective_values_last,
					"time": t1-t0}
	torch.save(recon_dict, "recon_dict.pt")
	os.remove(dump_name)
	gc.collect()
if __name__ == '__main__':
    reconstruction()