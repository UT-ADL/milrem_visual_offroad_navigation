import onnxruntime as ort

from models.global_planner import GlobalPlanner, normalize_probabilities


class GlobalPlannerOnnx(GlobalPlanner):
    def __init__(self, map_reader, config, model_path, convert_to_px=True):
        super(GlobalPlannerOnnx, self).__init__(map_reader, config, convert_to_px)

        self.forward_session = ort.InferenceSession(
            model_path,
            providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        )

    def predict_probabilities(self, current_position, goal_position):
        if self.convert_to_px:
            positions_pix = self.map_reader.lat_lon_to_pixel([current_position[0], goal_position[0]], [current_position[1], goal_position[1]])
            current_position = positions_pix[0]
            goal_position = positions_pix[1]

        masked_map = self.create_masked_map(current_position, goal_position)

        forward_model_in = {'masked_map': masked_map}
        forward_model_out = self.forward_session.run(None, forward_model_in)
        pred = forward_model_out[0].squeeze()
        self.probability_map = normalize_probabilities(pred)