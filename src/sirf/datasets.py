import torch
import sirf.STIR as pet
import os

class SIRF3DDataset():
    def __init__(self, base_path, name, tracer, count_level, realisation, num_subsets):
        if "thorax3D" == name:
            self.thorax(base_path, tracer, count_level, realisation, num_subsets)
        elif "brainweb3D" == name:
            self.brainweb(base_path, tracer, count_level, realisation, num_subsets)
        else:
            raise NotImplementedError("Dataset not implemented")

    def brainweb(self, base_path, tracer, count_level, realisation, num_subsets):
        if count_level == "low":
            count = "4e+07"
        elif count_level == "high":
            count = "2e+08"
        assert tracer in ["FDG", "Amyloid"], "must be FDG or Amyloid"
        bin_eff = pet.AcquisitionData(base_path + f"/{tracer}_bin_eff_hr.hs")   
        sensitivity_image_sirf = pet.ImageData(base_path + f"/{tracer}_sensitivity_image.hv")  
        osem = pet.ImageData(base_path + f"/noisy/{tracer}_osem_{count}_{realisation}.hv")
        self.osem = torch.tensor(osem.as_array()).to("cuda").float().unsqueeze(1)
        noisy_measurements_name = base_path + \
            f"/noisy/{tracer}_noisy_measurements_{count}_{realisation}.hs"
        
        measurements = pet.AcquisitionData(noisy_measurements_name)
        
        self.image_sirf = osem.get_uniform_copy(1.)
        views = measurements.shape[2]
        
        self.objectives_sirf = []
        for i in range(num_subsets):
            subset_idxs = list(range(views))[i:][::num_subsets]
            # Get subset of data
            noisy_measurements_sirf = measurements.get_subset(subset_idxs)
            bin_eff_subset = bin_eff.get_subset(subset_idxs)

            # SET UP THE ACQUISITION MODEL
            sensitivity_factors = pet.AcquisitionSensitivityModel(bin_eff_subset)
            acquisition_model = pet.AcquisitionModelUsingParallelproj()
            acquisition_model.set_acquisition_sensitivity(sensitivity_factors)
            acquisition_model.set_up(noisy_measurements_sirf, osem)
            
            objective_sirf = pet.make_Poisson_loglikelihood(noisy_measurements_sirf, acq_model = acquisition_model)
            objective_sirf.set_up(self.image_sirf)
            self.objectives_sirf.append(objective_sirf)
            print("Objective function", i+1, "of", num_subsets, "set up")
            
        
        self.sensitivity_image = torch.tensor(sensitivity_image_sirf.clone().as_array()).to("cuda").float().unsqueeze(1)
        self.fov = torch.zeros_like(self.sensitivity_image)
        self.fov[self.sensitivity_image!=0] = 1.

