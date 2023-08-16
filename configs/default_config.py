import ml_collections


def get_default_configs():

    config = ml_collections.ConfigDict()
    config.device = "cuda"
    config.seed = 1


    config.guided_p_uncond = 0.1 # 0.1 # None 

    config.normalisation = "image_scale"
    # sde configs
    config.sde = sde = ml_collections.ConfigDict()
    sde.type = "vesde" # "vpsde", "vesde" "heatdiffusion"

    # the largest noise scale sigma_max was choosen according to Technique 1 from [https://arxiv.org/pdf/2006.09011.pdf], 
    if sde.type == "vesde":
        sde.sigma_min = 0.01
        sde.sigma_max = 40. #for 40 vpsde, 0.1 for heatidffusion
    if sde.type == "vpsde":
        # only for vpsde
        sde.beta_min = 0.1
        sde.beta_max = 10

    if sde.type == "heatdiffusion":
        # used for HeatDiffusion
        sde.T_max = 64

    # training configs
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.epochs = 2000
    training.log_freq = 25
    training.lr = 1e-4
    training.ema_decay = 0.999
    training.ema_warm_start_steps = 50 # only start updating ema after this amount of steps 

    # validation configs
    config.validation = validation = ml_collections.ConfigDict()
    validation.batch_size = 8
    validation.snr = 0.05
    validation.num_steps = 500
    validation.eps = 1e-4
    validation.sample_freq = 0 #10

    # sampling configs 
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.batch_size = 1
    sampling.snr = 0.05
    sampling.num_steps = 1000 
    sampling.eps = 1e-4
    sampling.sampling_strategy = "predictor_corrector"
    sampling.start_time_step = 0

    sampling.load_model_from_path = "/localdata/AlexanderDenker/pet_score_based/checkpoints/version_02"
    sampling.model_name = "model.pt"


    # data configs - specify in other configs
    config.data = ml_collections.ConfigDict()
    config.data.im_size = 128 

    # forward operator config - specify in other configs
    config.forward_op = ml_collections.ConfigDict()

    # model configs
    config.model = model = ml_collections.ConfigDict()
    model.model_name = 'OpenAiUNetModel'
    if config.guided_p_uncond == None:
        model.in_channels = 1
    else:
        model.in_channels = 2
    model.model_channels = 64
    model.out_channels = 1
    model.num_res_blocks = 3
    model.attention_resolutions = [32, 16, 8]
    model.channel_mult = (1, 2, 2, 4, 4)
    model.conv_resample = True
    model.dims = 2
    model.num_heads = 4
    model.num_head_channels = -1
    model.num_heads_upsample = -1
    model.use_scale_shift_norm = True 
    model.resblock_updown = False
    model.use_new_attention_order = False
    model.max_period = 0.005


    return config