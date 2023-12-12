
import pytorch_lightning as pl

from gnm_train.visualizing.action_utils import visualize_traj_pred
from gnm_train.visualizing.distance_utils import visualize_dist_pred
from gnm_train.visualizing.visualize_utils import to_numpy


class BaseExperiment(pl.LightningModule):

    def __init__(self, project_folder, num_images_log, normalize):
        super(BaseExperiment, self).__init__()
        self.project_folder = project_folder
        self.num_images_log = num_images_log
        self.normalize = normalize

    def log_predictions(self, batch, dist_pred, action_pred, eval_type):

        (dist_obs_image, dist_goal_image, dist_trans_obs_image,
         dist_trans_goal_image, dist_label, dist_dataset_index) = batch["distance"]

        (action_obs_image, action_goal_image, action_trans_obs_image, action_trans_goal_image,
         action_goal_pos, action_label, action_dataset_index) = batch["action"]

        visualize_dist_pred(
            to_numpy(dist_obs_image),
            to_numpy(dist_goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            eval_type,
            self.project_folder,
            self.current_epoch,
            self.num_images_log,
            use_wandb=True,
            display=False
        )
        visualize_traj_pred(
            to_numpy(action_obs_image),
            to_numpy(action_goal_image),
            to_numpy(action_dataset_index),
            to_numpy(action_goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            eval_type,
            self.normalize,
            self.project_folder,
            self.current_epoch,
            self.num_images_log,
            use_wandb=True,
            display=False
        )