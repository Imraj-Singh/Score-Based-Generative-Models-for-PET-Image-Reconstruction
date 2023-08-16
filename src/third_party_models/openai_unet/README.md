### Open AI guided diffusion model (https://github.com/openai/guided-diffusion).

As used  **Diffusion Models Beat GANs on Image Synthesis** (https://arxiv.org/pdf/2105.05233.pdf).

I added output rescaling as used by Song so that the output is in the right range from the start of training.


Defaults for the UNet
res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )