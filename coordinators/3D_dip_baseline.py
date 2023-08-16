import matplotlib.pyplot as plt
import torch, sys, os, time
import sirf.STIR as pet
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.getcwd()))
from src import SIRF3DProjection, SIRF3DDataset
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
num_iterations = 11200
betas = [0.0,0.2,0.05,0.075,0.3]
max_norm = 1000
for tracer in ["FDG", "Amyloid"]:
    for realisation in range(5):
        dataset = SIRF3DDataset(base_path, name, tracer, count_level, realisation, num_subsets)
        dataset_name = f"{name}_{count_level}_{realisation}_{tracer}"
        check_folder_create("path_to/coordinators/DIP/", dataset_name)
        for beta in betas:
            projection = SIRF3DProjection(image_sirf = dataset.image_sirf.clone(), 
                    objectives_sirf = dataset.objectives_sirf,
                    sensitivity_image = dataset.sensitivity_image.clone(),
                    fov = dataset.fov.clone(), 
                    num_subsets = num_subsets, 
                    num_iterations = num_iterations)
            
            
            projection.set_beta(beta)
            print(f"beta: {projection.beta}")
            x_new = projection.get_DIP(path = f"path_to/coordinators/DIP/{dataset_name}/",
                                    x = dataset.osem.clone(), 
                                    beta = beta, 
                                    lr = 1e-3,
                                    max_norm = max_norm)
            
            check_folder_create(f"path_to/coordinators/DIP/{dataset_name}/", f"DIP_beta_{beta}")
            path = f"path_to/coordinators/DIP/{dataset_name}/DIP_beta_{beta}/"
            res_dict = {"objective_values": projection.objective_values,
                        "times": projection.times}
            torch.save(res_dict,path+"dict.pt")