import torch

from viz.base_visualizer import BaseVisualizer, BLUE
from viz.util_camera import to_camera_frame


class VAEVisualizer(BaseVisualizer):

    def __init__(self, goal_conditioning=False):
        super().__init__(goal_conditioning)
        self.sample_actions = None

    @torch.no_grad()
    def predict(self, model, obs_tensor, goal_tensor):
        predictions, _, _, _ = model(obs_tensor.unsqueeze(dim=0), self.goal_tensor.unsqueeze(dim=0))
        predicted_actions = predictions[1].detach().numpy()
        predicted_distance = predictions[0][0].item()

        distance, actions = model.sample(obs_tensor.unsqueeze(dim=0), 20)
        self.sample_actions = actions.detach().numpy()

        return predicted_actions, predicted_distance

    def draw_extras(self, img):
        if self.sample_actions is not None:
            for i in range(self.sample_actions.shape[0]):
                sampled_trajectory = self.sample_actions[i]
                to_camera_frame(img, sampled_trajectory, BLUE)
