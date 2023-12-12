import cv2
import numpy as np
import ros_numpy
from geometry_msgs.msg import Transform
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo

import yaml


def load_camera_transform(tf_base_link_to_cam_opt_frame_file):
    with open(tf_base_link_to_cam_opt_frame_file, 'r') as tf_yaml_file:
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


def load_camera_model(camera_info_config_file):
    with open(camera_info_config_file, 'r') as camera_info_yaml_file:
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


def to_camera_frame(rectified_image, tf_base_link_to_cam_opt_frame_file, camera_info_config_file, trajectory, color, radius=5):
    base_link_to_cam_frame_transform = load_camera_transform(tf_base_link_to_cam_opt_frame_file)
    relative_points = np.ones((trajectory.shape[0], 4))
    relative_points[:, 0:2] = trajectory[:, 0:2]
    # Only x, y coordinates are predicted, so there is no z value.
    # We expect trajectory to be on the flat plan and this arbitrary value looked nicest in visualisation.
    z = -0.5
    relative_points[:, 2] = z
    relative_points = np.concatenate([[[0, 0, z, 1.0]], relative_points])
    true_in_camera_frame = np.dot(base_link_to_cam_frame_transform, relative_points.T).T

    camera_pixel_coords = np.zeros(shape=(true_in_camera_frame.shape[0], 2))
    z = np.zeros(true_in_camera_frame.shape[0])

    camera_model = load_camera_model(camera_info_config_file)
    for id, point in enumerate(true_in_camera_frame):
        point3d = point[:3]
        z[id] = point3d[-1]
        camera_pixel_coords[id] = camera_model.project3dToPixel(point3d)

    for i, pixel_coord in enumerate(camera_pixel_coords):
        if i > 0:
            scaled_radius = int(np.abs((1 / (z[i]+0.001) * radius)))
            current_wp = (int(pixel_coord[0]), int(pixel_coord[1]))
            cv2.circle(rectified_image, current_wp, scaled_radius, color, 2)

            prev_wp = (int(camera_pixel_coords[i-1][0]), int(camera_pixel_coords[i-1][1]))
            cv2.line(rectified_image, prev_wp, current_wp, color, 2)
