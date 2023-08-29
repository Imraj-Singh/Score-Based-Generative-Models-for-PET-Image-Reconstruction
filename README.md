# Score-Based Generative Models for PET Image Reconstruction
Official code for [Score-Based Generative Models (SGM) for PET Image Reconstruction](https://arxiv.org/abs/2308.14190) (MELBA, under-review) by [Imraj RD Singh](https://www.imraj.dev/), [Alexander Denker](http://www.math.uni-bremen.de/zetem/cms/detail.php?template=parse_title&person=AlexanderDenker), [Riccardo Barbano](https://scholar.google.com/citations?user=6jYGiC0AAAAJ), [Željko Kereta](http://www0.cs.ucl.ac.uk/people/Z.Kereta.html), [Bangti Jin](https://www.math.cuhk.edu.hk/people/academic-staff/btjin), [Kris Thielemans](https://iris.ucl.ac.uk/iris/browse/profile?upi=KTHIE60), [Peter Maass](https://user.math.uni-bremen.de/pmaass/), [Simon Arridge](https://iris.ucl.ac.uk/iris/browse/profile?upi=SRARR14).

I. Singh, A. Denker and R. Barbano have equal contribution.

In this work we address PET-specific challenges such as; non-negativity of measurements/images, varying dynamic range of underlying radio-tracer distributions, and low-count Poisson noise on measurements requiring a Poisson Log-Likelihood (PLL). Further, we develop methods for 3D reconstruction, propose a guided variant with a Magnetic Resonance (MR) image, and accelerate the method using subsets.

Our modifications can be summarised with the following diagram:
![Alt text](/modifications.png)
Where the sections pertain to those in the [paper](tbd). The most appropriate reconstruction proposed, PET-variant of Decomposed Diffusion Sampling (PET-DDS; where DDS is proposed for MRI and CT [here](https://doi.org/10.48550/arXiv.2303.05754)), was extended to 3D and the reconstruction steps are illustrated below:

![Alt text](/diagram.png)

## Use of open-source repositories

The work presented develops and adopts code from various repositories, where specific contributions are indicated at the top of sources. The most important repositories include:
* [SGM sampling methods for inverse problems](https://github.com/educating-dip/score_based_model_baselines)
* [pyParallelProj for 2D experiments data generation](https://github.com/gschramm/pyparallelproj)
* [SIRF-exercises for 3D experiments data generation](https://github.com/SyneRBI/SIRF-Exercises)
* [Normalised supervised PET baselines](https://github.com/Imraj-Singh/pet_supervised_normalisation)
* [DIVal for supervised deep learning architectures and training scripts](https://github.com/jleuschn/dival)
* [Guided diffusion repository for the diffusion model architecture](https://github.com/openai/guided-diffusion)
* [Deep image prior comparison](https://github.com/educating-dip/pet_deep_image_prior)

We thank the authors of the aforementioned repositories for their open-source development and contributions.

## Datasets and Reproducibility

The results of this work are *in-silico* simulations of the [BrainWeb dataset](https://brainweb.bic.mni.mcgill.ca/), and all datasets are freely available for download/generation. For 2D work, and training the score-model, we use the dataset available [here](https://zenodo.org/record/4897350), which can be downloaded through [pyParalellProj](https://github.com/gschramm/pyparallelproj). For 3D work we use the dataset available here [here](https://github.com/casperdcl/brainweb).

Files for the generation of 2D data can be found in [src/brainweb_2d/](src/brainweb_2d/). For 3D data generation we provide a juypter notebook [src/sirf/brainweb_3D.ipynb](src/sirf/brainweb_3D.ipynb).

Training of the score-model requires running script [main_score_based_models_train.py](main_score_based_models_train.py). All experiments with reconstruction techniques can be found in [coordinators/](coordinators/), and all results can be processed with files in [results/](results/).

For reproducibility we provide a devcontainer utilising docker to containerise the development environment required for this work. The files are located in [.devcontainer/](.devcontainer/), these files use scripts to setup up conda environments where the environment is defined with files in [scripts/](scripts/), we provided full list of static dependencies in [req.txt](scripts/req.txt). Please note that this project requires [SIRF](https://github.com/SyneRBI/SIRF) for 3D work.

Trained score-models are available [TBD](zenodo somewhere), and results are available upon request.

## Citation
Arxiv bibtex:
```
@misc{singh2023scorebased,
      title={Score-Based Generative Models for PET Image Reconstruction}, 
      author={Imraj RD Singh and Alexander Denker and Riccardo Barbano and Željko Kereta and Bangti Jin and Kris Thielemans and Peter Maass and Simon Arridge},
      year={2023},
      eprint={2308.14190},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

