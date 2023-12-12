from model.nomad_onnx import NomadOnnx
from viz.base_visualizer import BaseVisualizer


class NomadOnnxVisualizer(BaseVisualizer):

    def __init__(self, model_path, goal_conditioning):
        super().__init__(goal_conditioning)
        self.nomad_onnx = NomadOnnx(model_path)

    def predict(self, model, obs_tensor, goal_tensor):
        obs_tensor = obs_tensor.unsqueeze(dim=0).numpy()
        goal_tensor = goal_tensor.unsqueeze(dim=0).numpy()

        predicted_dist, predicted_actions = self.nomad_onnx.predict(obs_tensor, goal_tensor)

        return predicted_actions, predicted_dist.item()
