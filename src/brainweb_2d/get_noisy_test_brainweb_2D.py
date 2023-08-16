import pyparallelproj.coincidences as coincidences
import pyparallelproj.subsets as subsets
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
import pyparallelproj.algorithms as algorithms
import cupy as xp
import cupyx.scipy.ndimage as ndi
from brainweb import BrainWebClean
import torch, os
from tqdm import tqdm

# Adapted from https://github.com/gschramm/pyparallelproj/blob/main/examples/00_projections_and_reconstruction/02_osem.py

if __name__ == "__main__":
    coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
        num_rings=1,
        sinogram_spatial_axis_order=coincidences.
        SinogramSpatialAxisOrder['RVP'],
        xp=xp)

    mu_projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                                (128, 128, 1), (-127, -127, 0),
                                                (2, 2, 2))
    projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                                (128, 128, 1), (-127, -127, 0),
                                                (2, 2, 2))
    res_model = resolution_models.GaussianImageBasedResolutionModel(
        (128, 128, 1), tuple(4.5 / (2.35 * x) for x in (2, 2, 2)), xp, ndi)

    projector.image_based_resolution_model = res_model

    subsetter = subsets.SingoramViewSubsetter(coincidence_descriptor, 34)
    projector.subsetter = subsetter
    xp.random.seed(42)
    n_realisations = 10
    bool_tumour = [True, False]
    trues_per_volumes = [100, 50, 10, 7.5, 5, 2.5]
    for tumour in bool_tumour:

        if tumour:
            dataset = BrainWebClean(path_to_files="path_to/clean/clean_test_tumour.pt")
        else:
            dataset = BrainWebClean(path_to_files="path_to/clean/clean_test.pt")

        for trues_per_volume in trues_per_volumes:
            osem_pts = []
            scaling_factor_pts = [] 
            noisy_data_pts = []
            contamination_pts = []
            attenuation_pts = []
            print(f"Trues per volume {trues_per_volume}, tumour {tumour}")
            for idx in tqdm(range(len(dataset))):
                osem_tmps = []
                scaling_factor_tmps = [] 
                noisy_data_tmps = []
                contamination_tmps = []
                attenuation_tmps = []
                y, mu, gt = dataset[idx]
                gt = xp.from_dlpack(gt.squeeze().unsqueeze(-1).to("cuda"))
                mu = xp.from_dlpack(mu.squeeze().unsqueeze(-1).to("cuda"))
                y = xp.from_dlpack(y.squeeze().to("cuda"))
                for _ in range(n_realisations):
                    # simulate the attenuation factors (exp(-fwd(attenuation_image)))
                    attenuation_factors = xp.exp(-mu_projector.forward(mu))
                    projector.multiplicative_corrections = attenuation_factors * 1. / 30

                    # scale the image such that we get a certain true count per emission voxel value
                    emission_volume = xp.where(gt > 0)[0].shape[0] * 8
                    current_trues_per_volume = float(y.sum() / emission_volume)

                    scaling_factor = (trues_per_volume / current_trues_per_volume)

                    image_fwd_scaled = y*scaling_factor

                    # simulate a constant background contamination
                    contamination_scale = image_fwd_scaled.mean()
                    contamination = xp.full(projector.output_shape,
                                            contamination_scale,
                                            dtype=xp.float32)

                    # generate noisy data
                    data = xp.random.poisson(image_fwd_scaled + contamination).astype(xp.uint16).astype(xp.float32)


                    reconstructor = algorithms.OSEM(data, contamination, projector, verbose=False)
                    reconstructor.run(1, evaluate_cost=False)

                    osem_x = reconstructor.x

                    osem_tmps.append(torch.from_dlpack(osem_x[:,:,0])[None].float().cuda())
                    scaling_factor_tmps.append(torch.tensor(scaling_factor)[None].float().cuda())
                    noisy_data_tmps.append(torch.from_dlpack(data)[None].float().cuda())
                    contamination_tmps.append(torch.tensor(contamination_scale)[None].float().cuda())
                    attenuation_tmps.append(torch.from_dlpack(attenuation_factors)[None].float().cuda())
                
                """ import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(osem_x[:,:,0].get())
                ax[1].imshow(mu[:,:,0].get())
                ax[2].imshow(gt[:,:,0].get())
                plt.show()
                exit() """
            
                osem_pts.append(torch.cat(osem_tmps)[None])
                scaling_factor_pts.append(torch.cat(scaling_factor_tmps)[None])
                noisy_data_pts.append(torch.cat(noisy_data_tmps)[None])
                contamination_pts.append(torch.cat(contamination_tmps)[None])
                attenuation_pts.append(torch.cat(attenuation_tmps)[None])

            osem_reconstruction = torch.cat(osem_pts)
            scaling_factor = torch.cat(scaling_factor_pts)
            noisy_data = torch.cat(noisy_data_pts)
            contamination_scales = torch.cat(contamination_pts)
            attenuation_factors = torch.cat(attenuation_pts)

            save_dict = {'osem': osem_reconstruction, 
                        'scale_factor': scaling_factor,
                        'measurements': noisy_data,
                        'contamination_factor': contamination_scales,
                        'attn_factors': attenuation_factors}
            if tumour:
                torch.save(save_dict, f"E:/projects/pet_score_model/src/brainweb_2d/noisy/noisy_test_tumour_{trues_per_volume}.pt")
            else:
                torch.save(save_dict, f"E:/projects/pet_score_model/src/brainweb_2d/noisy/noisy_test_{trues_per_volume}.pt")
            
            del osem_pts, scaling_factor_pts, noisy_data_pts, contamination_pts, attenuation_pts
            del osem_reconstruction, scaling_factor, noisy_data, contamination_scales, attenuation_factors
            del save_dict
