import numpy as np
import torch
from matplotlib import pyplot as plt

from data.dataset import TrajectoryDataset, OneHotEncodedDataset
from data.mapping import MapReader


def denorm(x, map_size):
    return x * map_size + map_size


def create_contrastive_figure(model, dataset_dir, map_path, map_size, trajectory_length,
                              trajectory_sampling_rate, trajectory_index, save_path):
    map_reader = MapReader(map_path, map_size)
    dataset = TrajectoryDataset(dataset_path=dataset_dir,
                                map_reader=map_reader,
                                map_size=map_size,
                                trajectory_min_len=trajectory_length,
                                trajectory_max_len=trajectory_length,
                                trajectory_sampling_rate=trajectory_sampling_rate)

    with torch.no_grad():
        trajectory, map_tensor, map_img = dataset[trajectory_index]
        map_tensor = map_tensor.unsqueeze(dim=0).to("cuda")
        map_features = model.encode_map(map_tensor)

        t = torch.linspace(-1.0, 1.0, 40)
        x, y = torch.meshgrid(t, t, indexing='ij')
        wps = torch.stack((x, y), dim=-1)

        neg_probs = np.zeros((wps.shape[0], wps.shape[1]))
        for i in range(wps.shape[0]):
            for j in range(wps.shape[1]):
                neg_coord = torch.cat([trajectory[0], trajectory[-1], wps[i, j]]).unsqueeze(dim=0).to("cuda")
                neg_probs[i, j] = model(neg_coord, map_features).item()

        wps_denorm = denorm(wps, map_size).view(-1, 2)
        neg_probs = neg_probs.flatten()
        neg_probs_filter = np.where(neg_probs > 0.01)
        neg_probs = neg_probs[neg_probs_filter]
        wps_denorm = wps_denorm[neg_probs_filter]

    fig = plt.figure(figsize=(7, 7))
    plt.scatter(wps_denorm[:, 0], wps_denorm[:, 1], c=neg_probs, cmap="Reds", alpha=0.6)
    plt.clim(0.0, 1.0)
    plt.colorbar()
    plt.scatter(denorm(trajectory[:, 0], map_size), denorm(trajectory[:, 1], map_size), color='blue', alpha=0.6)
    plt.imshow(map_img)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def create_segmentation_figure(model, dataset_dir, map_path, map_size, trajectory_length,
                              trajectory_sampling_rate, trajectory_index, save_path, config):
    map_reader = MapReader(map_path, map_size)
    dataset = OneHotEncodedDataset(dataset_path=dataset_dir,
                                   map_reader=map_reader,
                                   map_size=map_size,
                                   trajectory_min_len=trajectory_length,
                                   trajectory_max_len=trajectory_length,
                                   trajectory_sampling_rate=trajectory_sampling_rate,
                                   config=config)

    with torch.no_grad():
        masked_map, trajectory_mask, trajectory, map_img = dataset[trajectory_index]
        masked_map = torch.tensor(masked_map).to("cuda")

        fig, axs = plt.subplots(1, 3, figsize=(21, 7))
        axs[0].imshow(map_img)
        axs[0].set_title("Map")
        axs[0].scatter(trajectory[:, 0], trajectory[:, 1], color='blue', alpha=0.6, s=10)

        axs[1].imshow(trajectory_mask)
        axs[1].set_title("Target")

        masked_map = masked_map.unsqueeze(dim=0)
        pred = model(masked_map).squeeze().detach().cpu().numpy()
        axs[2].imshow(pred)
        axs[2].set_title("Predictions")

        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
