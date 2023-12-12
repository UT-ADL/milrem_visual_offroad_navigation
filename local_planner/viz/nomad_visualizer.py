import torch

from viz.base_visualizer import BaseVisualizer


class NomadVisualizer(BaseVisualizer):

    @torch.no_grad()
    def predict(self, model, obs_tensor, goal_tensor):
        predicted_dist, predicted_actions = model(obs_tensor.unsqueeze(dim=0),
                                                  goal_tensor.unsqueeze(dim=0),
                                                  self.goal_conditioning)
        return predicted_actions, predicted_dist.item()
