import torch
from glob import glob
import os, sys
sys.path.append("/home/user/sirf/")
from src import PSNR, SSIM
import sirf.STIR as pet

def get_crc_std_psnr_ssim(vols, tumours, backgrounds, ref, img_idx):
    crcs = []
    std_background = backgrounds.sum(dim=0, keepdim=True).repeat(len(vols),1,1,1)
    stds = vols[torch.nonzero(std_background, as_tuple=True)].std(dim = 0).mean().item()
    
    for i in range(len(tumours)):
        tumour_roi_idx = torch.nonzero(tumours[i], as_tuple=True)
        background_roi_idx = torch.nonzero(backgrounds[i], as_tuple=True)
        cr_refs = ref[tumour_roi_idx].mean() / ref[background_roi_idx].mean()
        crc_r = []
        for i in range(len(vols)):
            cr_est = vols[i][tumour_roi_idx].mean() / vols[i][background_roi_idx].mean()
            crc_r.append(((cr_est-1)/(cr_refs-1)).item())
        crcs.append(sum(crc_r)/len(crc_r))
    
    psnrs_r = []
    ssims_r = []
    for i in range(len(vols)):
        psnrs_r.append(PSNR((vols[i]).numpy(), (ref).numpy()))
        ssims_r.append(SSIM((vols[i]).numpy(), (ref).numpy()))
    return crcs, stds, sum(psnrs_r)/len(psnrs_r), sum(ssims_r)/len(ssims_r), vols[0,img_idx]

def get_sweep_mean_results(path, img_id=3):
    result = torch.load(path)
    datafit_strengths = []
    psnrs = []
    ssims = []
    crcs = []
    stds = []
    show_images = []
    for datafit_strength in result.keys():
        datafit_strengths.append(float(datafit_strength))
        psnrs.append(result[datafit_strength]["psnr"])
        ssims.append(result[datafit_strength]["ssim"])
        crcs.append(result[datafit_strength]["crc"])
        stds.append(result[datafit_strength]["std"])
        show_images.append(result[datafit_strength]["show_images"])
    psnrs = [x for _, x in sorted(zip(datafit_strengths, psnrs))]
    ssims = [x for _, x in sorted(zip(datafit_strengths, ssims))]
    crcs = [x for _, x in sorted(zip(datafit_strengths, crcs))]
    stds = [x for _, x in sorted(zip(datafit_strengths, stds))]
    datafit_strengths = sorted(datafit_strengths)
    return {"datafit_strengths": datafit_strengths, "psnr": psnrs, "ssim": ssims, "crc": crcs, "std": stds, "show_images": show_images}


def save_sweep_dicts(unique_swept_datafit_strengths):
    for name in unique_swept_datafit_strengths.keys():
        if "brainweb3D" in name:
            tumours = torch.load("/home/user/sirf/src/sirf/brainweb_3D/tumours.pt")
            backgrounds = torch.load("/home/user/sirf/src/sirf/brainweb_3D/backgrounds.pt")
            if "FDG" in name:
                ref = torch.from_numpy(pet.ImageData(f"/home/user/sirf/src/sirf/brainweb_3D/FDG_PET_lr.hv").as_array())
                # Scaled reference as linear model and scaled data
                ref = ref * 0.0018871473105862091 / 4.0024639524408965
            elif "Amyloid" in name:
                ref = torch.from_numpy(pet.ImageData(f"/home/user/sirf/src/sirf/brainweb_3D/Amyloid_PET_lr.hv").as_array())
                # Scaled reference as linear model and scaled data
                ref = ref * 0.0024918709984937458 / 4.004922937685068
            else: raise NotImplementedError("Not a valid tracer")
            img_idx = [12,31,47,55,71]
        sweep_dict = {}
        folder_name = name.split("_")[0] + "_" + name.split("_")[1] + "_" + name.split("_")[2]
        file_name = ""
        for i in name.split("_")[3:]:
            file_name +=  i + "_"
        file_name = file_name[:-1]
        if "dds_3D" in name:
            file_name += "_beta_" +  name.split("_")[-1]
        check_folder_create("3D_dicts/", folder_name)
        while len(unique_swept_datafit_strengths[name]) > 0:
            result_path = unique_swept_datafit_strengths[name][0]
            if "dds" in name:
                # is lambda not beta in this case
                beta = result_path.split("/")[-3].split("_")[-3]
            elif "DIP" in name:
                beta = result_path.split("/")[-2].split("_")[-1]
            else:
                _, _, beta = result_path.split("/")[-2].split("_")
            sweep_dict[str(beta)] = {}
            beta_paths = []
            # Get all the realisations
            for beta_path in unique_swept_datafit_strengths[name]:
                if "dds" in name:
                    # is lambda not beta in this case
                    b = beta_path.split("/")[-3].split("_")[-3]
                elif "DIP" in name:
                    b = beta_path.split("/")[-2].split("_")[-1]
                else:
                    _, _, b = beta_path.split("/")[-2].split("_")
                if b == beta:
                    beta_paths.append(beta_path)
            vols = []
            for b_p in beta_paths:
                unique_swept_datafit_strengths[name].remove(b_p)
                vols.append(torch.load(b_p))
            vols = torch.stack(vols)
            crc, std, psnr, ssim, show_images = get_crc_std_psnr_ssim(vols, tumours, backgrounds, ref, img_idx)
            sweep_dict[str(beta)]["crc"] = crc
            sweep_dict[str(beta)]["std"] = std
            sweep_dict[str(beta)]["psnr"] = psnr
            sweep_dict[str(beta)]["ssim"] = ssim
            sweep_dict[str(beta)]["show_images"] = show_images
        torch.save(sweep_dict, f"3D_dicts/{folder_name}/{file_name}.pt")


def check_folder_create(path, folder_name):
    CHECK_FOLDER = os.path.isdir(path+folder_name)
    if not CHECK_FOLDER:
        os.makedirs(path+folder_name)
        print("created folder : ", path+folder_name)

def get_unique_swept_datafit_strengths(base_path="/home/user/sirf/coordinators/FINAL_3D/**/volume.pt"):
    result_paths = glob(base_path, recursive=True)
    num_results = len(result_paths)
    # FIND ALL THE UNIQUE DATAFIT STRENGTH SWEEPS
    unique_swept_datafit_strengths = {}
    while len(result_paths) > 0:
        result_path = result_paths[0]
        if "dds_3D" in result_path:
            dataset = result_path.split("/")[-4]
            prior = result_path.split("/")[-3]
            data_name, count_level, _, tracer = dataset.split("_")
            prior_name = prior.split("_")[0]
            prior_beta = prior.split("_")[-1]
            name = f"{data_name}_{count_level}_{tracer}_{prior_name}_{prior_beta}"
        elif "DIP" in result_path:
            dataset = result_path.split("/")[-3]
            prior = result_path.split("/")[-2]
            prior_beta = prior.split("_")[2]
            data_name, count_level, _, tracer = dataset.split("_")
            prior_name = prior.split("_")[0]
            name = f"{data_name}_{count_level}_{tracer}_{prior_name}_{prior_beta}"
        else:
            dataset = result_path.split("/")[-3]
            prior = result_path.split("/")[-2]
            data_name, count_level, _, tracer = dataset.split("_")
            prior_name = prior.split("_")[0]
            name = f"{data_name}_{count_level}_{tracer}_{prior_name}"
        if name in unique_swept_datafit_strengths.keys():
            unique_swept_datafit_strengths[name].append(result_path)
        else:
            unique_swept_datafit_strengths[name] = []
            unique_swept_datafit_strengths[name].append(result_path)
        result_paths.remove(result_path)
    print(f"Altogether we have {num_results} results, and {len(unique_swept_datafit_strengths.keys())} individual sweeps.")
    return unique_swept_datafit_strengths


