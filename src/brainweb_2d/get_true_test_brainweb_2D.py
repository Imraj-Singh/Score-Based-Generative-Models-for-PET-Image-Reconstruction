import nibabel as nib
import pyparallelproj.coincidences as coincidences
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.resolution_models as resolution_models
from tqdm import tqdm
import cupy as xp
import torch, os
import cupyx.scipy.ndimage as ndi
from tumor_generator import Generate2DTumors


# Adapted from https://github.com/gschramm/pyparallelproj/blob/main/examples/00_projections_and_reconstruction/02_osem.py

if __name__=="__main__":
    coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
                                    num_rings=1,
                                    sinogram_spatial_axis_order=coincidences.
                                    SinogramSpatialAxisOrder['RVP'],
                                    xp=xp)
    
    mu_projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                                (256,256,1), 
                                                (-127.5, -127.5, 0),
                                                (1,1,2))
    
    true_projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                                (256,256,1), 
                                                (-127.5, -127.5, 0),
                                                (1,1,2))
    
    res_model = resolution_models.GaussianImageBasedResolutionModel((256,256,1), 
                tuple(4.5 / (2.35 * x) for x in (1,1,2)), xp, ndi)
    
    true_projector.image_based_resolution_model = res_model
    
    bool_tumour = [True, False]

    for tumour in bool_tumour:

        clean_data_pts = []
        mu_ref_pts = []
        image_ref_pts = []
        mri_ref_pts = []
        if tumour:
            background_pts = []
            tumour_rois_pts = []

        nii_pet = nib.as_closest_canonical(nib.load(f'path_to/examples/data/brainweb_petmr/subject04/sim_0/true_pet.nii.gz'))
        nii_mu = nib.as_closest_canonical(nib.load(f'path_to/examples/data/brainweb_petmr/subject04/mu.nii.gz'))
        nii_mri = nib.as_closest_canonical(nib.load(f'path_to/examples/data/brainweb_petmr/subject04/t1.nii.gz'))

        # pet image resolution [1,1,2] mm
        image_gt = xp.array(nii_pet.get_fdata(), dtype=xp.float32)
        image_gt = (image_gt[:, :, ::2] + image_gt[:, :, 1::2])/2

        # pet image resolution [2,2,2] mm
        image_ref = (image_gt[::2, :, :] + image_gt[1::2, :, :])/2
        image_ref = (image_ref[:, ::2, :] + image_ref[:, 1::2, :])/2

        # mu image resolution [1,1,2] mm
        mu_gt = xp.array(nii_mu.get_fdata(), dtype=xp.float32)
        mu_gt = (mu_gt[:, :, ::2] + mu_gt[:, :, 1::2]) /2

        # mu image resolution [2,2,2] mm
        mu_ref = (mu_gt[::2, :, :] + mu_gt[1::2, :, :])/2
        mu_ref = (mu_ref[:, ::2, :] + mu_ref[:, 1::2, :])/2

        # mri image resolution [1,1,2] mm
        mri_gt = xp.array(nii_mri.get_fdata(), dtype=xp.float32)
        mri_gt = (mri_gt[:, :, ::2] + mri_gt[:, :, 1::2]) /2

        # mri image resolution [2,2,2] mm
        mri_ref = (mri_gt[::2, :, :] + mri_gt[1::2, :, :])/2
        mri_ref = (mri_ref[:, ::2, :] + mri_ref[:, 1::2, :])/2

        for slice_number in range(image_ref.shape[-1]):
            # ENSURE THERE ARE AT LEAST 2000 NON-ZERO PIXELS IN SLICE
            if len(xp.nonzero(image_ref[:, :, [slice_number]])[0]) > 2000:
                attenuation_factors = xp.exp(-mu_projector.forward(mu_gt[:, :, [slice_number]]))
                true_projector.multiplicative_corrections = attenuation_factors * 1./30
                if tumour:
                    image_gt_slice, background, tumour_rois = Generate2DTumors(xp.asnumpy(image_gt[:, :, slice_number]))
                    image_gt_slice = xp.expand_dims(xp.asarray(image_gt_slice),-1)
                    image_ref_slice = (image_gt_slice[::2, :, :] + image_gt_slice[1::2, :, :])/2
                    image_ref_slice = (image_ref_slice[:, ::2, :] + image_ref_slice[:, 1::2, :])/2
                    tumour_rois = (tumour_rois[:, ::2, :] + tumour_rois[:, 1::2, :])/2
                    tumour_rois = (tumour_rois[:, :, ::2] + tumour_rois[:, :, 1::2])/2
                    tumour_rois[tumour_rois < 1.] = 0.
                    background = (background[::2, :] + background[1::2, :])/2
                    background = (background[:, ::2] + background[:, 1::2])/2
                    background[background < 1.] = 0.
                    background_pts.append(torch.from_numpy(background)[None][None].float().cuda())
                    tumour_rois_pts.append(torch.from_numpy(tumour_rois)[None].float().cuda())
                    #print("Has tumour")
                else:
                    image_gt_slice = image_gt[..., [slice_number]]
                    image_ref_slice = image_ref[..., [slice_number]]
                    #print("Has no tumour")

                clean_data = true_projector.forward(image_gt_slice)
                clean_data_pts.append(torch.from_dlpack(clean_data[None][None]).float().cuda())
                mu_ref_pts.append(torch.from_dlpack(mu_ref[:, :, slice_number][None][None]).float().cuda())
                image_ref_pts.append(torch.from_dlpack(image_ref_slice[:, :, 0][None][None]).float().cuda())
                mri_ref_pts.append(torch.from_dlpack(mri_ref[:, :, slice_number][None][None]).float().cuda())

        clean_data_pts = torch.cat(clean_data_pts)
        mu_ref_pts = torch.cat(mu_ref_pts)
        image_ref_pts = torch.cat(image_ref_pts)
        mri_ref_pts = torch.cat(mri_ref_pts)
        if tumour:
            background_pts = torch.cat(background_pts)
            tumour_rois_pts = torch.cat(tumour_rois_pts)
            recon_dict = {'clean_measurements': clean_data_pts, 'mu': mu_ref_pts, 'reference': image_ref_pts, 'background': background_pts, 'tumour_rois': tumour_rois_pts, 'mri': mri_ref_pts}
            torch.save(recon_dict, "path_to/clean/clean_test_tumour.pt")
        else:
            recon_dict = {'clean_measurements': clean_data_pts, 'mu': mu_ref_pts, 'reference': image_ref_pts, 'mri': mri_ref_pts}
            torch.save(recon_dict, "path_to/clean/clean_test.pt")