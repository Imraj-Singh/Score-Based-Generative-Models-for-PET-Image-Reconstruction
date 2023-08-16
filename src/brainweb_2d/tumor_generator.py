import numpy as np
from skimage.morphology import area_closing, isotropic_erosion, isotropic_dilation
from skimage.draw import ellipse
from skimage.filters import gaussian


def Generate2DTumors(image):
    seg = np.zeros_like(image)
    foreground, background = 1, 0
    seg[image <= 0] = background
    seg[image > 0] = foreground
    seg = isotropic_erosion(area_closing(seg, 1e4), 20)

    n_ellipses = np.random.choice([1,1,1,2,2,3])
    # (r, c, r_radius, c_radius, shape=None, rotation=0.0)
    shape = image.shape
    tumour_rois = []
    background = np.zeros_like(image)
    for n_ellipse in range(n_ellipses):
        r_radius = max(np.random.poisson(10.),4)
        c_radius = max(np.random.poisson(10.),4)

        tmp_seg = isotropic_erosion(seg, max(r_radius, c_radius))
        tmp_coords = np.where(tmp_seg == 1)
        if tmp_coords[0].shape[0] == 0:
            break
        tmp_coord_idx = np.random.randint(0, tmp_coords[0].shape[0])
        rr_tmp, cc_tmp = ellipse(tmp_coords[0][tmp_coord_idx],
                                 tmp_coords[1][tmp_coord_idx], 
                                 r_radius, 
                                 c_radius, 
                                 shape=shape, 
                                 rotation=np.random.uniform(0, 2*np.pi))
        
        tumour = np.zeros_like(image)
        tumour[rr_tmp, cc_tmp] = 1.
        tumour_rois.append(tumour)

        intensity_factor = image.max()*np.random.uniform(1.3, 1.8)
        tmp_tumour  = tumour * intensity_factor
        tmp_tumour = gaussian(tmp_tumour, sigma=np.random.uniform(1., 3.0))

        image = np.maximum(tmp_tumour, image)

        tmp_background = isotropic_dilation(tumour, 20)*1.
        tmp_background *= seg
        background += tmp_background
    while len(tumour_rois) < 3:
        tumour_rois.append(np.zeros_like(image))
    background[background > 0] = 1.
    for n_ellipse in range(n_ellipses):
        tmp_rmv = isotropic_dilation(tumour_rois[n_ellipse], 8)*1.
        background *= -1*(tmp_rmv-1)
    return image, background, np.stack(tumour_rois)




if __name__=="__main__":
    import torch
    import matplotlib.pyplot as plt
    clean = torch.load("path_to/examples/data/clean/test_subset_clean.pt")
    image_clean = clean['reference'][2].squeeze().detach().numpy()
    from skimage.transform import resize
    image_clean = resize(image_clean, (image_clean.shape[0] * 2, image_clean.shape[1] * 2),
                       anti_aliasing=True)
    image, background, tumour_rois = Generate2DTumors(image_clean)
    print(tumour_rois.shape)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    add = np.zeros_like(image)
    for tumour in tumour_rois:
        add += tumour
    fig.colorbar(ax[0].imshow(image), ax=ax[0])
    fig.colorbar(ax[1].imshow(image_clean + add*3 + background*3), ax=ax[1])
    fig.colorbar(ax[2].imshow(image - image_clean + image.max()*background), ax=ax[2])
    plt.show()