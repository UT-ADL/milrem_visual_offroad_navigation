import onnxruntime as ort

from viz.base_visualizer import BaseVisualizer


class OnnxVisualizer(BaseVisualizer):

    def predict(self, model, obs_tensor, goal_tensor):
        forward_session = ort.InferenceSession(  # TODO: move to init
            model,
            providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        )

        forward_model_in = {
            'obs_img': obs_tensor.unsqueeze(dim=0).numpy(),
            'goal_img': goal_tensor.unsqueeze(dim=0).numpy()
        }
        forward_model_out = forward_session.run(None, forward_model_in)

        predicted_distance = forward_model_out[0][0].item()
        predicted_actions = forward_model_out[1]
        return predicted_actions, predicted_distance
