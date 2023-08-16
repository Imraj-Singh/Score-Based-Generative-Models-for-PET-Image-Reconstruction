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
    dataset = BrainWebClean(path_to_files="path_to/clean/clean_train.pt", mri=True)
    n_validation = 8
    validation_indices = torch.randperm(len(dataset))[:n_validation]
    dataset = torch.utils.data.Subset(dataset, validation_indices)
    trues_per_volumes = [2.5, 5, 7.5, 10, 50, 100]

    for trues_per_volume in trues_per_volumes:
        osem_pts = []
        scaling_factor_pts = [] 
        noisy_data_pts = []
        contamination_pts = []
        attenuation_pts = []
        print(f"Trues per volume {trues_per_volume}")
        for idx in tqdm(range(len(dataset))):
            y, mu, gt, _ = dataset[idx]
            gt = xp.from_dlpack(gt.cuda().squeeze().unsqueeze(-1))
            mu = xp.from_dlpack(mu.cuda().squeeze().unsqueeze(-1))
            y = xp.from_dlpack(y.cuda().squeeze())
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

            osem_pts.append(torch.from_dlpack(osem_x[:,:,0])[None][None].float().cpu())
            scaling_factor_pts.append(torch.tensor(scaling_factor)[None][None].float().cpu())
            noisy_data_pts.append(torch.from_dlpack(data)[None][None].float().cpu())
            contamination_pts.append(torch.tensor(contamination_scale)[None][None].float().cpu())
            attenuation_pts.append(torch.from_dlpack(attenuation_factors)[None][None].float().cpu())

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
        torch.save(save_dict, f"path_to/noisy/noisy_validation_{trues_per_volume}.pt")
        del osem_pts, scaling_factor_pts, noisy_data_pts, contamination_pts, attenuation_pts
        del osem_reconstruction, scaling_factor, noisy_data, contamination_scales, attenuation_factors
        del save_dict
    
    ys = []
    mus = []
    gts = []
    mris = []
    for i in range(n_validation):
        y, mu, gt, mri = dataset[i]
        ys.append(y)
        mus.append(mu)
        gts.append(gt)
        mris.append(mri)
    ys = torch.stack(ys)
    mus = torch.stack(mus)
    gts = torch.stack(gts)
    mris = torch.stack(mris)
    print(gts.shape)
    print(validation_indices)
    save_dict = {'clean_measurements': ys, 'mu': mus, 'reference': gts, 'mri': mris}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 8)
    for i in range(n_validation):
        ax[i].imshow(gts[i].squeeze())
    plt.show()
    torch.save(save_dict, f"path_to/clean/clean_validation.pt")
