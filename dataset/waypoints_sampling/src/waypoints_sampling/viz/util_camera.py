import cv2

import numpy as np
np.float = np.float64

import ros_numpy
from geometry_msgs.msg import Transform
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

import yaml


def load_camera_transform():
    with open("config/tf_base_link_to_cam_opt_frame.yaml", 'r') as tf_yaml_file:
        transform =yaml.load(tf_yaml_file, Loader=yaml.SafeLoader)

    tf_base_link_to_opt_frame = Transform()

    tf_base_link_to_opt_frame.translation.x = transform['translation']['x']
    tf_base_link_to_opt_frame.translation.y = transform['translation']['y']
    tf_base_link_to_opt_frame.translation.z = transform['translation']['z']
    tf_base_link_to_opt_frame.rotation.x = transform['rotation']['x']
    tf_base_link_to_opt_frame.rotation.y = transform['rotation']['y']
    tf_base_link_to_opt_frame.rotation.z = transform['rotation']['z']
    tf_base_link_to_opt_frame.rotation.w = transform['rotation']['w']
    tf_base_link_to_opt_frame_matrix = ros_numpy.numpify(tf_base_link_to_opt_frame)
    return tf_base_link_to_opt_frame_matrix


def load_camera_model():
    with open("config/camera_info.yaml", 'r') as camera_info_yaml_file:
        camera_info = yaml.load(camera_info_yaml_file, Loader=yaml.SafeLoader)

    camera_model = PinholeCameraModel()
    camera_info_msg = CameraInfo()

    camera_info_msg.header.seq = camera_info['header']['seq']
    camera_info_msg.header.stamp.secs = camera_info['header']['stamp']['secs']
    camera_info_msg.header.stamp.nsecs = camera_info['header']['stamp']['nsecs']
    camera_info_msg.header.frame_id = camera_info['header']['frame_id']

    camera_info_msg.height = camera_info['height']
    camera_info_msg.width = camera_info['width']

    camera_info_msg.distortion_model = camera_info['distortion_model']

    camera_info_msg.D = camera_info['D']
    camera_info_msg.K = camera_info['K']
    camera_info_msg.R = camera_info['R']
    camera_info_msg.P = camera_info['P']

    camera_info_msg.binning_x = camera_info['binning_x']
    camera_info_msg.binning_y = camera_info['binning_y']

    camera_info_msg.roi.x_offset = camera_info['roi']['x_offset']
    camera_info_msg.roi.y_offset = camera_info['roi']['y_offset']

    camera_info_msg.roi.height = camera_info['roi']['height']
    camera_info_msg.roi.width = camera_info['roi']['width']

    camera_info_msg.roi.do_rectify = camera_info['roi']['do_rectify']

    camera_model.fromCameraInfo(camera_info_msg)
    return camera_model


def to_camera_frame(rectified_image, trajectory, color):
    base_link_to_cam_frame_transform = load_camera_transform()
    relative_points = np.ones((trajectory.shape[0], 4))
    relative_points[:, 0:2] = trajectory[:, 0:2]
    # Only x, y coordinates are predicted, so there is no z value.
    # We expect trajectory to be on the flat plan and this arbitrary value looked nicest in visualisation.
    relative_points[:, 2] = -0.3
    true_in_camera_frame = np.dot(base_link_to_cam_frame_transform, relative_points.T).T

    camera_pixel_coords = np.zeros(shape=(true_in_camera_frame.shape[0], 2))
    z = np.zeros(true_in_camera_frame.shape[0])

    camera_model = load_camera_model()
    for id, point in enumerate(true_in_camera_frame):
        point3d = point[:3]
        z[id] = point3d[-1]
        camera_pixel_coords[id] = camera_model.project3dToPixel(point3d)

    for id, pixel in enumerate(camera_pixel_coords):
        center = (int(pixel[0]), int(pixel[1]))
        scaled_radius = int(np.abs((1 / (z[id]+0.001) * 5)))
        cv2.circle(rectified_image, center, scaled_radius, color, 2)
    
    with open("src/waypoints_sampling/src/config/vae.yaml", 'r') as vae_yaml_file:
        vae_params = yaml.safe_load(vae_yaml_file)

    step_size = vae_params['model_params']['pred_trajectory_length']
    num_samples = vae_params['model_params']['num_samples']

    if len(camera_pixel_coords) == step_size:
        for i in range(step_size-1):
            cv2.line(rectified_image, 
                        tuple(camera_pixel_coords[i].astype(int)), tuple(camera_pixel_coords[i+1].astype(int)), 
                        (0, 0, 255), 
                        4)
    else:
        trajectories = camera_pixel_coords.reshape(num_samples, step_size, 2)
        colors = [(0, 0, 0),
                  (255, 255, 255),
                  (255, 0, 0),
                  (0, 255, 0),
                  (255, 255, 0),
                  (0, 255, 255),
                  (255, 0, 255),
                  (192, 192, 192),
                  (128, 128, 128),
                  (128, 0, 0),
                  (128, 128, 0),
                  (0, 128, 0),
                  (128, 0, 128),
                  (0, 128, 128),
                  (0, 0, 128),
                  (0, 100, 0),
                  (0, 255, 0),
                  (46, 139, 87),
                  (175, 238, 238),
                  (245, 222, 179)]
        
        for id, traj in enumerate(trajectories):
            line_color = colors[id]
            # line_color = (255, 0, 0)
            for i in range(len(traj)-1):
                cv2.line(rectified_image,
                         tuple(traj[i].astype(int)), tuple(traj[i+1].astype(int)),
                         line_color,
                         1)