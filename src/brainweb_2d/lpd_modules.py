import torch
import os
import cupy as xp

# Loosely based on https://github.com/educating-dip/pet_deep_image_prior/blob/main/src/deep_image_prior/torch_wrapper.py

class LPDForwardFunction2D(torch.autograd.Function):
    """PET forward projection function

    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, projector, attn_factors):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.projector = projector
        ctx.attn_factors = attn_factors

        x_inp = x.detach()
        # convert pytorch input tensor into cupy array
        cp_x = xp.ascontiguousarray(xp.from_dlpack(x_inp))

        # a custom function that maps from cupy array to cupy array
        batch, channels = cp_x.shape[:2]
        b_y = []
        for sample in range(batch):
            projector.multiplicative_corrections = xp.from_dlpack(attn_factors[sample, 0, ...])
            c_y = []
            for channel in range(channels):
                c_y.append(projector.forward(cp_x[sample, channel, :, :][:, :, None]))
            b_y.append(xp.stack(c_y))
        # convert torch array to cupy array
        return torch.from_dlpack(xp.stack(b_y))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        if grad_output is None:
            return None, None, None
        else:
            projector = ctx.projector
            attn_factors = ctx.attn_factors

            cp_y = xp.from_dlpack(grad_output.detach())

            # a custom function that maps from cupy array to cupy array
            batch, channels = cp_y.shape[:2]
            b_x = []
            for sample in range(batch):
                projector.multiplicative_corrections = xp.from_dlpack(attn_factors[sample, 0, ...])
                c_x = []
                for channel in range(channels):
                    c_x.append(projector.adjoint(cp_y[sample, channel, :])[..., 0])
                b_x.append(xp.stack(c_x))
            b_x = xp.stack(b_x)
            # convert torch array to cupy array
            return torch.from_dlpack(b_x), None, None



class LPDAdjointFunction2D(torch.autograd.Function):
    """PET forward projection function

    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, y, projector, attn_factors):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.projector = projector
        ctx.attn_factors = attn_factors


        # convert pytorch input tensor into cupy array
        cp_y = xp.ascontiguousarray(xp.from_dlpack(y.detach()))

        # a custom function that maps from cupy array to cupy array
        batch, channels = cp_y.shape[:2]
        b_x = []
        for sample in range(batch):
            projector.multiplicative_corrections = xp.from_dlpack(attn_factors[sample, 0, ...])
            c_x = []
            for channel in range(channels):
                c_x.append(projector.adjoint(cp_y[sample, channel, :])[..., 0])
            b_x.append(xp.stack(c_x))
        b_x = xp.stack(b_x)
        # convert torch array to cupy array
        return torch.from_dlpack(b_x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        if grad_output is None:
            return None, None, None
        else:
            projector = ctx.projector
            attn_factors = ctx.attn_factors

            cp_x = xp.from_dlpack(grad_output.detach())

            # a custom function that maps from cupy array to cupy array
            batch, channels = cp_x.shape[:2]
            b_y = []
            for sample in range(batch):
                projector.multiplicative_corrections = xp.from_dlpack(attn_factors[sample, 0, ...])
                c_y = []
                for channel in range(channels):
                    c_y.append(projector.forward(cp_x[sample, channel, :, :][:, :, None]))
                b_y.append(xp.stack(c_y))
            b_y = xp.stack(b_y)
            # convert torch array to cupy array
            return torch.from_dlpack(b_y), None, None



if __name__ == "__main__":
    import os
    from pyparallelproj import petprojectors, coincidences, resolution_models
    import cupyx.scipy.ndimage as ndi
    from brainweb import BrainWebOSEM
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    detector_efficiency = 1./30
    coincidence_descriptor = coincidences.GEDiscoveryMICoincidenceDescriptor(
        num_rings=1,
        sinogram_spatial_axis_order=coincidences.
        SinogramSpatialAxisOrder['RVP'],
        xp=xp)

    projector = petprojectors.PETJosephProjector(coincidence_descriptor,
                                                (128, 128, 1), (-127, -127, 0),
                                                (2., 2., 2.))

    res_model = resolution_models.GaussianImageBasedResolutionModel(
        (128, 128, 1), tuple(4.5 / (2.35 * x) for x in (2., 2., 2.)), xp, ndi)
    
    projector.image_based_resolution_model = res_model

    dataset_list = []
    dataset = BrainWebOSEM(part="test", noise_level=5)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    batch_size = 10
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    sinogram = torch.nn.Parameter(torch.ones(batch_size, 1, 112880,
                                    device=device))
    image = torch.nn.Parameter(torch.ones(batch_size, 1,
                                    *projector._image_shape[:-1],
                                    device=device))
    optimizer = torch.optim.Adam([sinogram, image], lr=1e0)

    batch = next(iter(train_dl))
    reference = batch[0]
    reference = reference.to(device)

    scale_factor = batch[1]
    scale_factor = scale_factor.to(device)

    osem = batch[2]
    osem = osem.to(device)

    norm = batch[3]
    norm = norm.to(device)

    measurements = batch[4]
    measurements = measurements.to(device)
    contamination_factor = batch[5]
    contamination_factor = contamination_factor.to(device)

    attn_factors = batch[6]
    attn_factors = attn_factors.to(device)*detector_efficiency

    for epoch in tqdm(range(100)):
        optimizer.zero_grad()

        y_bar = LPDForwardFunction2D.apply(image, projector, attn_factors)
        loss = ((y_bar - measurements)**2).sum()
        loss.backward()
        optimizer.step()
        if epoch==0:
            print(loss.item())
    print(loss.item())
    fig, ax = plt.subplots(1, 2)
    print(image.max(), image.min())
    ax[0].imshow(reference[0, 0, ...].detach().cpu().numpy())
    ax[1].imshow(y_bar[0, 0, ...].detach().cpu().numpy())
    plt.show()





