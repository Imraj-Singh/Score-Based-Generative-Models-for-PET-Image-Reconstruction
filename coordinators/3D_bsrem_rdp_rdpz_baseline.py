import matplotlib.pyplot as plt
import torch, sys, os, time
import sirf.STIR as pet
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.getcwd()))
from src import herman_meyer_order, SIRF3DProjection, SIRF3DDataset
pet.set_verbosity(0)
pet.AcquisitionData.set_storage_scheme("memory")
#pet.MessageRedirector(info=None, warn=None, errr=None)

def check_folder_create(path, folder_name):
    CHECK_FOLDER = os.path.isdir(path+folder_name)
    if not CHECK_FOLDER:
        os.makedirs(path+folder_name)
        print("created folder : ", path+folder_name)


# name: brainweb3D
# count_level: low
# realisation: 0
# tracer: FDG
# base_path: /home/user/sirf/src/sirf/brainweb_3D

# name: thorax3D
# count_level: low
# realisation: 0
# tracer: 0
# base_path: /home/user/sirf/D690XCATnonTOF

# GET THE DATA
# measurements_subsets_sirf, acquisition_models_sirf, sensitivity_image, fov, image_sirf, osem, measurements_subsets

name = "brainweb3D"
count_level = "low"
realisation = "0"
tracer = "FDG"
base_path = "path_to/src/sirf/brainweb_3D"
num_subsets = 28
num_iterations = 2800

for tracer in ["FDG", "Amyloid"]:
    for realisation in range(5):
        dataset = SIRF3DDataset(base_path, name, tracer, count_level, realisation, num_subsets)
        dataset_name = f"{name}_{count_level}_{realisation}_{tracer}"
        check_folder_create("path_to/coordinators/BSREM/", dataset_name)
        for prior in ["rdpz", "rdp"]:
            if prior == "rdp":
                betas = [0.5,0.767,1.18,1.81,2.77,4.25,6.52,10.]
            elif prior == "rdpz":
                betas = [10.,15.3,23.5,36.1,55.4,85.,130.,200.]
            for beta in betas:
                projection = SIRF3DProjection(image_sirf = dataset.image_sirf.clone(), 
                        objectives_sirf = dataset.objectives_sirf,
                        sensitivity_image = dataset.sensitivity_image.clone(),
                        fov = dataset.fov.clone(), 
                        num_subsets = num_subsets, 
                        num_iterations = num_iterations)
                prior_name = f"{prior}_beta_{beta}"
                check_folder_create(f"path_to/coordinators/BSREM/{dataset_name}/", prior_name)
                full_path = f"path_to/coordinators/BSREM/{dataset_name}/{prior_name}/"
                projection.set_beta(beta)
                print(f"beta: {projection.beta}")
                x_new = projection.get_bsrem(full_path, dataset.osem.clone(), eta = 0.1, prior = prior, image_diff_tol = 1e-5)
                plt.plot(projection.objective_values)
                plt.title("Objective value")
                plt.savefig(f"{full_path}/objective.png", dpi=300, bbox_inches="tight")
                plt.close()
                    
                torch.save(x_new.squeeze().cpu(), f"{full_path}/volume.pt")