import numpy as np
import onnxruntime as ort

from waypoint_planner.model.noise_scheduler import DDPMScheduler
from waypoint_planner.model.nomad_util import get_action


class NomadOnnx:
    def __init__(self, model_path, num_samples=10, len_traj_pred=8, num_diffusion_iters=10):
        self.num_samples = num_samples
        self.len_traj_pred = len_traj_pred
        self.num_diffusion_iters = num_diffusion_iters

        self.encoder_session = create_onnx_session(f"{model_path}/encoder.onnx")
        self.dist_session = create_onnx_session(f"{model_path}/distance.onnx")
        self.action_session = create_onnx_session(f"{model_path}/action.onnx")

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,

        )

    def predict(self, obs_tensor, goal_tensor):
        vision_features = self.encode_features(goal_tensor, obs_tensor)
        predicted_distance = self.predict_distance(vision_features)
        predicted_actions = self.predict_actions(vision_features)
        return predicted_distance, predicted_actions

    def encode_features(self, goal_tensor, obs_tensor):
        mask = np.zeros(1, dtype=int).repeat(goal_tensor.shape[0])
        obs_tensor = np.repeat(obs_tensor, repeats=goal_tensor.shape[0], axis=0)
        encoder_in = {
            'obs_img': obs_tensor,
            'goal_img': goal_tensor,
            'input_goal_mask': mask
        }
        encoder_out = self.encoder_session.run(None, encoder_in)

        # (B, obs_horizon * obs_dim)
        obs_cond = np.repeat(encoder_out[0], repeats=self.num_samples, axis=0)
        return obs_cond

    def predict_distance(self, vision_features):
        dist_in = {
            'vision_features': vision_features
        }
        dist_out = self.dist_session.run(None, dist_in)
        return dist_out[0][0]  # take first, in our case they should be all the same as goal image is same

    def predict_actions(self, vision_features):
        # initialize action from Gaussian noise
        naction = np.random.randn(self.num_samples, self.len_traj_pred, 2).astype(np.float32)
        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        for k in self.noise_scheduler.timesteps[:]:
            # predict noise
            action_in = {
                'sample': naction,
                'timestep': np.array(k),
                'global_cond': vision_features
            }
            action_out = self.action_session.run(None, action_in)
            noise_pred = action_out[0]

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            )
        predicted_actions = get_action(naction).squeeze()
        return predicted_actions


def create_onnx_session(model_path):
    session = ort.InferenceSession(
        model_path,
        providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    )
    return session
