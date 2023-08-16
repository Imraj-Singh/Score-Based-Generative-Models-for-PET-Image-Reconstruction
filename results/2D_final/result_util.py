import torch
from glob import glob
import os, sys
sys.path.append("/home/user/sirf/")
from src import PSNR, SSIM
import numpy as np

def get_std_crc_ssim_psnr(result):
    images = result["images"]
    refs = result["ref"].unsqueeze(1)
    if "lesion_rois" in result.keys():
        if result["lesion_rois"].shape[-1] == result["ref"].shape[-1]:
            lesion_rois = result["lesion_rois"]
            background_rois = result["background_rois"]
        else:
            lesion_rois = torch.zeros_like(refs)
            background_rois = torch.zeros_like(refs)
            lesion_rois[refs!=0] = 1
            background_rois[refs!=0] = 1
    else:
        lesion_rois = torch.zeros_like(refs)
        background_rois = torch.zeros_like(refs)
        lesion_rois[refs!=0] = 1
        background_rois[refs!=0] = 1
    psnrs = []
    ssims = []
    crcs = []
    stds = []
    for img_idx in range(images.shape[0]):
        image = images[img_idx].cpu().numpy()
        ref = refs[img_idx].squeeze().cpu().numpy()
        lesion_roi = lesion_rois[img_idx].cpu().numpy()
        background_roi = background_rois[img_idx].squeeze().cpu().numpy()
        psnr_r = []
        ssim_r = []
        crc_r = []
        b_bar_r = []
        for realisation in image:
            psnr_r.append(torch.asarray(PSNR(realisation,ref)))
            ssim_r.append(torch.asarray(SSIM(realisation, ref)))
            if background_roi.sum() != 0:
                background_idx = np.nonzero(background_roi)
                b_bar = realisation[background_idx]
                b_t = ref[background_idx]
                crc_t = []
                for i in range(len(lesion_roi)):
                    if lesion_roi[i,:,:].sum() != 0:
                        tumour_roi_idx = np.nonzero(lesion_roi[i,:,:])
                        a_bar = realisation[tumour_roi_idx]
                        a_t = ref[tumour_roi_idx]
                        if a_bar.mean() == 0 and b_bar.mean() == 0:
                            crc_t.append(np.array([0.0]))
                        else:
                            crc_t.append((a_bar.mean()/b_bar.mean() - 1) / (a_t.mean()/b_t.mean() - 1))
                crc_r.append(torch.asarray(crc_t).mean())
                b_bar_r.append(torch.asarray(b_bar))
        std = (torch.std(torch.stack(b_bar_r), dim=0)/torch.clamp(torch.stack(b_bar_r).mean(0),1e-9)).mean()
        psnrs.append(torch.asarray(psnr_r))
        ssims.append(torch.asarray(ssim_r))
        crcs.append(torch.asarray(crc_r))
        stds.append(torch.asarray(std))
    return  torch.stack(psnrs), torch.stack(ssims), torch.stack(crcs), torch.stack(stds)

def get_sweep_mean_results(path, img_id=3):
    result = torch.load(path)
    datafit_strengths = []
    kldivs = []
    psnrs = []
    ssims = []
    crcs = []
    stds = []
    show_images = []
    for datafit_strength in result.keys():
        kldiv = []
        psnr = []
        ssim = []
        crc = []
        std = []
        images = []
        datafit_strengths.append(float(datafit_strength))
        for image_num in result[datafit_strength].keys():
            kldiv.append(result[datafit_strength][image_num]["kldiv"])
            psnr.append(result[datafit_strength][image_num]["psnr"])
            ssim.append(result[datafit_strength][image_num]["ssim"])
            crc.append(result[datafit_strength][image_num]["crc"])
            std.append(result[datafit_strength][image_num]["std"])
            images.append(result[datafit_strength][image_num]["images"])
        kldivs.append(sum(kldiv) / len(kldiv))
        psnrs.append(sum(psnr) / len(psnr))
        ssims.append(sum(ssim) / len(ssim))
        crcs.append(sum(crc) / len(crc))
        stds.append(sum(std) / len(std))
        show_images.append(images[img_id])
    kldivs = [x for _, x in sorted(zip(datafit_strengths, kldivs))]
    psnrs = [x for _, x in sorted(zip(datafit_strengths, psnrs))]
    ssims = [x for _, x in sorted(zip(datafit_strengths, ssims))]
    crcs = [x for _, x in sorted(zip(datafit_strengths, crcs))]
    stds = [x for _, x in sorted(zip(datafit_strengths, stds))]
    datafit_strengths = sorted(datafit_strengths)
    return {"datafit_strengths": datafit_strengths, "kldivs": kldivs, "psnrs": psnrs, "ssims": ssims, "crcs": crcs, "stds": stds, "show_images": show_images}


def get_individual_dict(result):
    psnr, ssim, crc, std = get_std_crc_ssim_psnr(result)
    n_images = len(result["ref"])
    individual_dict = {}
    for i in range(n_images):
        individual_dict[str(i)] = {}
        # Mean accross realisations
        individual_dict[str(i)]["kldiv"] = result["kldiv"][i].mean()
        individual_dict[str(i)]["psnr"] = psnr[i].mean()
        individual_dict[str(i)]["ssim"] = ssim[i].mean()
        individual_dict[str(i)]["crc"] = crc[i].mean()
        # Std accross realisations
        individual_dict[str(i)]["std"] = std[i]
        # save the first realisation
        individual_dict[str(i)]["images"] = result["images"][i][0]
    return individual_dict

def save_sweep_dicts(unique_swept_datafit_strengths):
    for name in unique_swept_datafit_strengths.keys():
        sweep_dict = {}
        for result_path in unique_swept_datafit_strengths[name]:
            result = torch.load(result_path)
            if "naive" in result_path or "dps" in result_path:
                datafit_strength = result["penalty"]
            elif "dds" in result_path:
                if "osem_num_epochs" in result_path:
                    datafit_strength = result["num_epochs"]
                elif "anchor_num_epochs" in result_path:
                    datafit_strength = result["beta"]
            elif "rdp" in result_path:
                datafit_strength = result["beta"]
            else:
                raise NotImplementedError
            individual_dict = get_individual_dict(result)
            sweep_dict[str(datafit_strength)] = individual_dict
        if "tumour" in name:
            name = name.replace("tumour_","")
            torch.save(sweep_dict, f"tumour/{name}.pt")
        else:
            torch.save(sweep_dict, f"non_tumour/{name}.pt")

def get_unique_swept_datafit_strengths(base_path="E:/projects/pet_score_model/coordinators/SBM_2/**/*.pt"):
    result_paths = glob(base_path, recursive=True)
    num_results = len(result_paths)
    # FIND ALL THE UNIQUE DATAFIT STRENGTH SWEEPS
    unique_swept_datafit_strengths = {}
    while len(result_paths) > 0:
        result_path = result_paths[0]
        result = torch.load(result_path)
        identifers = []
        if "tumour" in result_path:
            identifers.append("tumour")
        if "vesde" in result_path:
            identifers.append("vesde")
        if "vpsde" in result_path:
            identifers.append("vpsde")
        if "OSEMNLL_" in result_path:
            identifers.append("OSEMNLL")
        if "dps" in result_path:
            identifers.append("dps")
        if "naive" in result_path:
            identifers.append("naive")
        if "dds" in result_path:
            identifers.append("dds")
            if "osem_num_epochs" in result_path:
                identifers.append("osem")
            elif "anchor_num_epochs" in result_path:
                identifers.append("anchor")
                identifers.append("epoch_" + str(result["num_epochs"]))
            else: raise NotImplementedError("DDS needs to be osem or anchor")
        if "guided" in result_path:
            identifers.append("guided")
            identifers.append("gstrength_" + str(result["gstrength"]))
        if "rdp" in result_path:
            identifers.append("rdp")
        name = ""
        for identifer in identifers:
            name += identifer + "_"
        if name in unique_swept_datafit_strengths.keys():
            unique_swept_datafit_strengths[name].append(result_path)
        else:
            unique_swept_datafit_strengths[name] = []
            unique_swept_datafit_strengths[name].append(result_path)
        result_paths.remove(result_path)
    print(f"Altogether we have {num_results} results, and {len(unique_swept_datafit_strengths.keys())} individual sweeps.")
    return unique_swept_datafit_strengths


