import os
import numpy as np 
from datetime import datetime 
import yaml 
from torch.utils.data import DataLoader

from src import (get_standard_sde, get_standard_score, BrainWebScoreTrain, 
			score_model_simple_trainer)



#from configs.ellipses_configs import get_config
from configs.default_config import get_default_configs

def coordinator():

	config = get_default_configs()

	if config.guided_p_uncond is not None:
		print("Train Guided Score Model")
		assert config.model.in_channels == 2, "input channels = 2 for guided model"

	sde = get_standard_sde(config)
	score_model = get_standard_score(config, sde, use_ema=True, load_model=False)

	brain_dataset = BrainWebScoreTrain(base_path="/localdata/AlexanderDenker/pet_data/"  ,#"E:/projects/pet_score_model/src/brainweb_2d/", 
										guided= True if config.guided_p_uncond is not None else False,
										normalisation = config.normalisation)
	
	train_dl = DataLoader(brain_dataset,batch_size=config.training.batch_size, num_workers=6)

	print(f" # Parameters: {sum([p.numel() for p in score_model.parameters()]) }")
	today = datetime.now()
	
	if config.guided_p_uncond is not None:
		log_dir = '/localdata/AlexanderDenker/pet_score_based/guided/' + config.sde.type

	else:
		log_dir = '/localdata/AlexanderDenker/pet_score_based/' + config.sde.type

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	found_version = False 
	version_num = 0
	while not found_version:
		if os.path.isdir(os.path.join(log_dir, "version_" + str(version_num))):
			version_num += 1
		else:
			found_version = True 

	log_dir = os.path.join(log_dir, "version_" + str(version_num))
	os.makedirs(log_dir)

	with open(os.path.join(log_dir,'report.yaml'), 'w') as file:
		yaml.dump(config, file)

	score_model_simple_trainer(
		score=score_model.to(config.device),
		sde=sde, 
		train_dl=train_dl,
		optim_kwargs={
			'epochs': config.training.epochs,
			'lr': config.training.lr,
			'ema_warm_start_steps': config.training.ema_warm_start_steps,
			'log_freq': config.training.log_freq,
			'ema_decay': config.training.ema_decay
			},
			val_kwargs={
			'batch_size': config.validation.batch_size,
			'num_steps': config.validation.num_steps,
			'snr': config.validation.snr,
			'eps': config.validation.eps,
			'sample_freq' : config.validation.sample_freq
			},
		device=config.device,
		log_dir=log_dir,
		guided_p_uncond=config.guided_p_uncond
	)


if __name__ == '__main__': 
	coordinator()