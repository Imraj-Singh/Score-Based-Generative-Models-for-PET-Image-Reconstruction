import torch

if __name__=="__main__":

    names = ["test", "test_tumour"]
    noise_levels = ["_2.5", "_5", "_7.5", "_10", "_50", "_100"]
    samples = [5,15,25,35,45,55,65,75]

    for name in names:
        for noise in noise_levels:
            data = torch.load(f"path_to/noisy/noisy_{name+noise}.pt")
            subset_data = {}
            for key in data.keys():
                subset_data[key] = data[key][samples, ...]
            torch.save(subset_data, f"path_to/noisy/noisy_subset_{name+noise}.pt")

    for name in names:
        data = torch.load(f"path_to/clean/clean_{name}.pt")
        subset_data = {}
        for key in data.keys():
            subset_data[key] = data[key][samples, ...]
        torch.save(subset_data, f"path_to/clean/clean_subset_{name}.pt")