from models.global_planner import GlobalPlanner, normalize_probabilities
from models.util import load_model


class GlobalPlannerTorch(GlobalPlanner):

    def __init__(self, map_reader, config, model_path, convert_to_px=True):
        super(GlobalPlannerTorch, self).__init__(map_reader, config, convert_to_px)
        self.model = load_model(model_path)

    def predict_probabilities(self, current_position, goal_position):
        masked_map = self.create_masked_map(current_position, goal_position)
        pred = self.model(masked_map).squeeze().detach().numpy()
        self.probability_map = normalize_probabilities(pred)