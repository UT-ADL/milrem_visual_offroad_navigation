import numpy as np

ACTION_STATS = {
    "min": np.array([-2.5, -4]),  # [min_dx, min_dy]
    "max": np.array([5, 4])  # [max_dx, max_dy]
}


def get_action(diffusion_output):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    ndeltas = unnormalize_data(ndeltas, ACTION_STATS)
    actions = np.cumsum(ndeltas, axis=1)
    return actions


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data