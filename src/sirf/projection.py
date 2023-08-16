from .herman_meyer import herman_meyer_order
import torch, time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from .dip import PETUNet
import sirf.STIR as pet

class SIRF3DProjection():
    def __init__(self, image_sirf, objectives_sirf, sensitivity_image, fov, 
                 num_subsets=1, num_iterations=None):
        self.image_sirf = image_sirf
        self.objectives_sirf = objectives_sirf
        # Need to detach to avoid memory leak
        self.sensitivity_image = sensitivity_image.detach()
        self.fov = fov.detach()

        self.num_subsets = num_subsets
        self.num_iterations = num_iterations
        self.access_order = herman_meyer_order(num_subsets)
        self.global_iteration = 0
        self.lambd = None
        self.beta = None
        self.objective_values = []
        self.objective_values_last = []
        self.pll_values = []
        self.pll_values_last = []
        self.recons_last = []

    def set_lambd(self, lambd):
        self.lambd = lambd

    def set_beta(self, beta):
        self.beta = beta

    def get_pll_grad(self, x):
        self.image_sirf.fill(x.detach().cpu().numpy().squeeze())
        tmp = self.objectives_sirf[self.subset_num()].get_gradient(self.image_sirf)
        return self.num_subsets * torch.from_numpy(tmp.as_array()).float().to(x.device).unsqueeze(1)

    def get_pll_value(self, x):
        self.image_sirf.fill(x.detach().cpu().numpy().squeeze())
        return - self.num_subsets * self.objectives_sirf[self.subset_num()].get_value(self.image_sirf)

    def rdpz_rolled_components(self, x):
        neighbours = [1,-1]
        x_neighbours = x.clone().repeat(1,2,1,1)
        for i in range(2):
            x_neighbours[:,[i]] = torch.roll(x, shifts=(neighbours[i]), dims=(0))
        x_neighbours[0,:] = x[1]
        x_neighbours[-1,:] = x[-2]
        return x_neighbours

    def get_rdpz_value(self, x):
        x_neighbours = self.rdpz_rolled_components(x)
        x = x.repeat(1,2,1,1)
        numerator = (x - x_neighbours)**2
        denominator = (x + x_neighbours + 2 * torch.abs(x - x_neighbours))
        rdp = - (numerator/denominator).sum(dim=1, keepdim=True)
        rdp[numerator[:,[0]] < 1e-9] = 0
        rdp[numerator[:,[1]] < 1e-9] = 0
        rdp[self.fov==0] = 0
        return rdp.sum()

    def get_rdpz_grad(self, x):
        x_neighbours = self.rdpz_rolled_components(x)
        x = x.repeat(1,2,1,1)
        numerator = (x - x_neighbours)*(2 * torch.abs(x-x_neighbours) + x + 3 * x_neighbours)
        denominator = (x + x_neighbours + 2 * torch.abs(x - x_neighbours))**2
        gradient = - (numerator/denominator)
        gradient[denominator < 1e-9] = 0
        gradient = gradient.sum(dim=1, keepdim=True)
        gradient[self.fov==0] = 0
        return gradient

    def get_rdp_value(self, x):
        rdp = torch.zeros_like(x)
        # we want relected in axial
        x_neighbours = torch.nn.functional.pad(x.swapaxes(0,1), (1,1,1,1,1,1), mode='reflect').swapaxes(0,1)
        for i in [0,1,2]:
            for j in [0,1,2]:
                for k in [0,1,2]:
                    i_end = x_neighbours.shape[0] - (2 - i)
                    j_end = x_neighbours.shape[2] - (2 - j)
                    k_end = x_neighbours.shape[3] - (2 - k)
                    numerator = (x - x_neighbours[i:i_end,:,j:j_end,k:k_end])**2
                    denominator = (x + x_neighbours[i:i_end,:,j:j_end,k:k_end] + \
                                   2 * torch.abs(x - x_neighbours[i:i_end,:,j:j_end,k:k_end]))
                    rdp -= (numerator/torch.clamp(denominator, 1e-9))
        rdp[self.fov==0] = 0
        return rdp.sum()

    def get_rdp_grad(self, x):
        gradient = torch.zeros_like(x)
        # we want relected in axial
        x_neighbours = torch.nn.functional.pad(x.swapaxes(0,1), (1,1,1,1,1,1), mode='reflect').swapaxes(0,1)
        for i in [0,1,2]:
            for j in [0,1,2]:
                for k in [0,1,2]:
                    i_end = x_neighbours.shape[0] - (2 - i)
                    j_end = x_neighbours.shape[2] - (2 - j)
                    k_end = x_neighbours.shape[3] - (2 - k)
                    numerator = (x - x_neighbours[i:i_end,:,j:j_end,k:k_end]) * \
                        (2 * torch.abs(x-x_neighbours[i:i_end,:,j:j_end,k:k_end]) + \
                         x + 3 * x_neighbours[i:i_end,:,j:j_end,k:k_end])
                    denominator = (x + x_neighbours[i:i_end,:,j:j_end,k:k_end] + \
                        2 * torch.abs(x - x_neighbours[i:i_end,:,j:j_end,k:k_end]))**2
                    gradient -=  (numerator/torch.clamp(denominator,1e-9))
        gradient[self.fov==0] = 0
        return gradient
    
    def get_anchor_value(self, x):
        return ((self.x_anchor-x).detach()**2).sum()

    def get_anchor_grad(self, x):
        tmp = (self.x_anchor-x).detach()
        tmp[self.fov==0] = 0
        return tmp

    def subset_num(self):
        return self.access_order[self.global_iteration%self.num_subsets]

    def get_anchor(self, x, scale_factor):
        x = torch.clamp(x, 0).swapaxes(2,3).flip(2,3)
        x[self.fov==0] = 0
        self.x_anchor = (scale_factor*x).clone().detach()
        x_old = (scale_factor*x).clone().detach()
        del x
        for _ in range(self.num_iterations):
            t0 = time.time()
            preconditioner = torch.clamp(x_old, 1e-4)/self.sensitivity_image
            preconditioner[self.fov==0] = 0.
            pll_value = self.get_pll_value(x_old)
            pll_grad = self.get_pll_grad(x_old)
            anchor_value = self.get_anchor_value(x_old)
            anchor_grad = self.get_anchor_grad(x_old)
            gradient = pll_grad + self.lambd * anchor_grad
            x_new = torch.clamp(x_old + preconditioner * gradient, 0)
            x_new[self.fov==0] = 0
            x_old = x_new
            self.pll_values.append(pll_value)
            self.objective_values.append((pll_value + self.lambd * anchor_value).item())
            self.global_iteration += 1
            t1 = time.time()
            #print("Iteration: ", self.global_iteration,". Objective value", self.objective_values[-1], ". Time taken: ", t1-t0, "s", ". Subset: ", self.subset_num(), ". x_old sum: ", x_old.sum())
        self.pll_values_last.append(self.pll_values[-1])
        self.objective_values_last.append(self.objective_values[-1])
        return torch.nan_to_num(x_old.swapaxes(2,3).flip(2,3)/scale_factor, nan=0, posinf=0, neginf=0), scale_factor
    
    def get_anchor_rdpz(self, x, scale_factor):
        x = torch.clamp(x, 0).swapaxes(2,3).flip(2,3)
        x[self.fov==0] = 0
        self.x_anchor = (scale_factor*x).clone().detach()
        x_old = (scale_factor*x).clone().detach()
        del x
        for _ in range(self.num_iterations):
            t0 = time.time()
            preconditioner = torch.clamp(x_old, 1e-4)/self.sensitivity_image
            preconditioner[self.fov==0] = 0.
            pll_value = self.get_pll_value(x_old)
            pll_grad = self.get_pll_grad(x_old)
            anchor_value = self.get_anchor_value(x_old)
            anchor_grad = self.get_anchor_grad(x_old)
            rdpz_value = self.get_rdpz_value(x_old)
            rdpz_grad = self.get_rdpz_grad(x_old)
            gradient = pll_grad + self.lambd * anchor_grad + self.beta * rdpz_grad
            x_new = torch.clamp(x_old.detach() + preconditioner * gradient, 0)
            x_new[self.fov==0] = 0
            x_old = x_new
            self.pll_values.append(pll_value)
            self.objective_values.append((pll_value + self.lambd * anchor_value + self.beta * rdpz_value).item())
            self.global_iteration += 1
            t1 = time.time()
            #print("Iteration: ", self.global_iteration,". Objective value", self.objective_values[-1], ". Time taken: ", t1-t0, "s", ". Subset: ", self.subset_num())
        self.pll_values_last.append(self.pll_values[-1])
        self.objective_values_last.append(self.objective_values[-1])
        return torch.nan_to_num(x_old.swapaxes(2,3).flip(2,3)/scale_factor, nan=0, posinf=0, neginf=0), scale_factor

    def get_bsrem(self, folder, x, eta = 0.04, prior = "rdp", image_diff_tol = 4.0):
        if self.beta is None:
            raise ValueError("beta not set")
        if prior == "rdp":
            get_prior_value = self.get_rdp_value
            get_prior_grad = self.get_rdp_grad
        if prior == "rdpz":
            get_prior_value = self.get_rdpz_value
            get_prior_grad = self.get_rdpz_grad
        x = torch.clamp(x, 0)
        x_old = x.clone().detach()
        for _ in range(self.num_iterations):
            epoch = self.global_iteration//self.num_subsets
            alpha = 1/(eta*epoch+1)
            t0 = time.time()
            preconditioner = (x_old + 1e-9)/self.sensitivity_image
            preconditioner[self.fov==0] = 0.
            pll_grad = self.get_pll_grad(x_old)
            prior_value = get_prior_value(x_old)
            prior_grad = get_prior_grad(x_old)
            gradient = pll_grad + self.beta * prior_grad
            x_new = torch.clamp(x_old.detach() + alpha * preconditioner * gradient, 0)
            x_new[self.fov==0] = 0
            image_diff_norm = ((x_new - x_old).pow(2)/x_new)[x_new!=0].mean()
            x_old = x_new
            t1 = time.time()
            if self.global_iteration%self.num_subsets == 0:
                pll_value = self.get_pll_value(x_old)
                self.pll_values.append(pll_value)
                self.objective_values.append((pll_value + self.beta * prior_value).item())
                print("Iter: ", self.global_iteration, \
                    ". Obj val", self.objective_values[-1], \
                        ". Time: ", t1-t0, "s", \
                        ". Subset: ", self.subset_num(), \
                            ". Img norm: ", x_new.sum().item(), \
                                ". Img diff norm: ", image_diff_norm.item())
            if image_diff_norm.item() < image_diff_tol:
                print(f"Converged within image difference norm of {image_diff_tol}")
                break
            self.global_iteration += 1
        plt.imshow(x_old.detach().cpu().numpy()[30,0])
        plt.colorbar()
        plt.savefig(f"{folder}/epoch_{epoch}.png", dpi=300, bbox_inches="tight")
        plt.close()
        return x_new

    def get_DIP(self, path, x, beta, lr, max_norm=1):
        torch.manual_seed(42)
        input_img = torch.randn_like(x).swapaxes(0,1).unsqueeze(0)
        network = PETUNet(ch = 16, size = input_img.shape[2:]).to(x.device)
        optimiser = torch.optim.Adam(network.parameters(), lr)
        prior = pet.RelativeDifferencePrior()
        prior.set_penalisation_factor(beta)
        prior.set_gamma(1.0)
        prior.set_up(self.image_sirf)
        for i in range(len(self.objectives_sirf)):
            self.objectives_sirf[i].set_prior(prior)
            self.objectives_sirf[i].set_up(self.image_sirf)
        self.times = []
        for _ in range(self.num_iterations + 1):
            t0 = time.time()
            optimiser.zero_grad()
            reconstruction = network(input_img)
            loss = - ObjectiveFunctionModule3D.apply(reconstruction, 
                                                     self.image_sirf, 
                                                     self.objectives_sirf[self.subset_num()])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm = max_norm)
            optimiser.step()
            reconstruction = reconstruction.squeeze().unsqueeze(1).detach()
            reconstruction[self.fov==0] = 0
            self.times.append(time.time()-t0)
            self.objective_values.append(loss.item())
            if self.global_iteration % 100 == 0:
                check_folder_create(path, f"DIP_beta_{beta}_iters_{self.global_iteration}")
                torch.save(reconstruction.squeeze().cpu(), \
                           f"{path}DIP_beta_{beta}_iters_{self.global_iteration}/volume.pt")
            self.global_iteration += 1

        return reconstruction.squeeze().unsqueeze(1)
    
import os

def check_folder_create(path, folder_name):
    CHECK_FOLDER = os.path.isdir(path+folder_name)
    if not CHECK_FOLDER:
        os.makedirs(path+folder_name)
        print("created folder : ", path+folder_name)


class ObjectiveFunctionModule3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, image_template, sirf_obj):
        ctx.device = x.device
        ctx.sirf_obj = sirf_obj
        ctx.image_template = image_template
        ctx.x = x.detach().cpu().numpy().squeeze()
        ctx.x = ctx.image_template.fill(ctx.x)
        value_np = ctx.sirf_obj.get_value(ctx.x)
        return torch.tensor(value_np).to(ctx.device)

    @staticmethod
    def backward(ctx, in_grad):
        grads_np = ctx.sirf_obj.get_gradient(ctx.x).as_array()
        grads = torch.from_numpy(grads_np).to(ctx.device) * in_grad
        return grads[None,None,...], None, None, None