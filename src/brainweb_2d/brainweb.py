import torch 


class BrainWebClean(torch.utils.data.Dataset):
    def __init__(self, path_to_files="path_to/test_dict.pt", mri=False):

        self.path_to_files = path_to_files
        self.data = torch.load(path_to_files, map_location=torch.device('cpu'))
        self.mri = mri
    
    def __len__(self):
        return self.data['clean_measurements'].shape[0]

    def __getitem__(self, idx):
        y = self.data["clean_measurements"][idx, ...]
        mu = self.data["mu"][idx, ...]
        gt = self.data["reference"][idx, ...]
        if self.mri:
            mri = self.data["mri"][idx, ...]
            return y, mu, gt, mri
        return y, mu, gt

class BrainWebOSEM(torch.utils.data.Dataset):
    def __init__(self, part, noise_level, base_path="path_to/src/brainweb_2d/", static_path = None, device="cpu", guided=False):
        assert noise_level in [2.5, 5, 7.5, 10, 50, 100, "2.5", "5", "7.5", "10", "50", "100"], "noise level has to be 2.5, 5, 7.5, 10, 50, 100"
        assert part in ["train", "test", "test_tumour", "subset_test_tumour", "subset_test", "validation"], 'part has to be "train", "test", "test_tumour", "subset_test_tumour", "subset_test", "validation"'

        self.part = part 
        self.noise_level = noise_level
        self.guided = guided

        self.base_path = base_path
        # dict_keys(['osem', 'scale_factor', 'measurements', 'contamination_factor', 'attn_factors'])
        self.noisy = torch.load(base_path+"noisy/noisy_"+ self.part + "_" + str(noise_level)+".pt", map_location=torch.device(device))
        # dict_keys(['clean_measurements', 'mu', 'reference']) 
        self.clean = torch.load(base_path+"clean/clean_"+part+".pt", map_location=torch.device(device))
        if static_path is not None:
            # dict_keys(['osem', 'scale_factor', 'measurements', 'contamination_factor', 'attn_factors'])
            self.noisy = torch.load(static_path, map_location=torch.device(device))
            # dict_keys(['clean_measurements', 'mu', 'reference']) 
            self.clean = torch.load(base_path+"clean/"+part+"_clean.pt", map_location=torch.device(device))
        if "tumour" in part:
            self.tumour = True
        else:
            self.tumour = False
    def __len__(self):
        return self.clean["reference"].shape[0]

    def __getitem__(self, idx):
        
        reference = self.clean["reference"][idx, ...].float()
        scale_factor = self.noisy["scale_factor"][idx]

        reference = reference*scale_factor[[0]] 

        if self.guided:
            reference = torch.cat((reference, self.clean["mri"][idx, ...].float()), dim=0)
        osem = self.noisy["osem"][idx, ...].float()


        norm = 1

        measurements = self.noisy["measurements"][idx, ...].float()
        contamination_factor = self.noisy["contamination_factor"][idx]
        attn_factors = self.noisy["attn_factors"][idx, ...].float()
        
        if self.part == "subset_test":
            measurements = measurements
            contamination_factor = contamination_factor
            attn_factors = attn_factors
            osem = osem

        if norm == 0:
            norm = torch.ones_like(norm)
            osem = torch.zeros_like(osem)
            reference = torch.zeros_like(reference)
            attn_factors = torch.ones_like(attn_factors)

        if self.tumour:
            background = self.clean["background"][idx, ...].float()
            tumour_rois = self.clean["tumour_rois"][idx, ...].float()
            return reference, scale_factor, osem, norm, measurements, contamination_factor, attn_factors, background, tumour_rois
        return reference, scale_factor, osem, norm, measurements, contamination_factor, attn_factors


class BrainWebSupervisedTrain(torch.utils.data.Dataset):
    def __init__(self, noise_level, base_path="path_to/pyparallelproj/examples/data/", device="cpu", guided=False):
        assert noise_level in [5, 10, 50, "5", "10", "50"], "noise level has to be 5, 10, 50"
        self.base_path = base_path

        # dict_keys(['clean_measurements', 'mu', 'reference', 'mri]) 
        clean = torch.load(base_path+"clean/clean_train.pt", map_location=torch.device(device))
        # dict_keys(['osem', 'scale_factor', 'measurements', 'contamination_factor', 'attn_factors'])
        self.noisy = torch.load(base_path+"noisy/noisy_train_"+str(noise_level)+".pt", map_location=torch.device(device))
        self.reference = clean["reference"]
        self.guided = guided
        if self.guided:
            self.mri = clean["mri"]

    def __len__(self):
        return self.reference

    def __getitem__(self, idx):
        reference = self.reference[idx,...].float()*self.noisy["scale_factor"][idx]

        osem = self.noisy["osem"][idx, ...].float()

        measurements = self.noisy["measurements"][idx, ...].float()

        contamination_factor = self.noisy["contamination_factor"][idx]

        attn_factors = self.noisy["attn_factors"][idx, ...].float()
        
        if self.guided:
            mri = self.mri[idx,...].float()
            return reference, mri, osem, measurements, contamination_factor, attn_factors

        return reference, osem, measurements, contamination_factor, attn_factors



class BrainWebScoreTrain(torch.utils.data.Dataset):
    def __init__(self, base_path="path_to/pyparallelproj/examples/data/", device="cpu", guided=False, normalisation="data_scale"):

        self.base_path = base_path
        # dict_keys(['clean_measurements', 'mu', 'reference', 'mri]) 
        clean = torch.load(base_path+"clean/clean_train.pt", map_location=torch.device(device))
        print(clean.keys())
        self.reference = clean["reference"]
        self.clean_measurements = clean["clean_measurements"]
        self.mri = clean["mri"]
        self.guided = guided

        self.normalisation = normalisation

    def __len__(self):
        return self.reference.shape[0]

    def __getitem__(self, idx):
        reference = self.reference[idx, ...].float()

        if self.normalisation == "data_scale":
            emission_volume = torch.where(reference > 0)[0].shape[0] * 8 # 2 x 2 x 2
            current_trues_per_volume = float(self.clean_measurements[idx].sum() / emission_volume)
        elif self.normalisation == "image_scale":
            emission_volume = torch.where(reference > 0)[0].shape[0]
            current_trues_per_volume = float(reference.sum() / emission_volume)
        else:
            raise NotImplementedError

        reference = reference/current_trues_per_volume 

        reference = reference* (0.5 + torch.rand(1))

        mri = self.mri[idx, ...].float()
        if self.guided:
            return torch.cat((reference, mri), dim=0)
        return reference

    
if __name__ == "__main__":

    dataset = BrainWebScoreTrain(base_path="path_to/", normalisation="image_scale")
    import matplotlib.pyplot as plt 
    import numpy as np 

    for i in range(10):
        batch = dataset[i]
        print(batch.min(), batch.max())