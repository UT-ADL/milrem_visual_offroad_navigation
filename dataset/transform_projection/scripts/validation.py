import os
import numpy as np
import yaml
import cv2

from image_geometry import PinholeCameraModel
from geometry_msgs.msg import Pose, Transform
from sensor_msgs.msg import CameraInfo

np.float = np.float64
import ros_numpy

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--tf_yaml_file", help="yaml file for static tf between base_link and camera_optical_frame")
parser.add_argument("--camera_info_yaml_file", help="yaml file for the camera intrinsics in camera_info ros msg")
parser.add_argument("--csv_file", help="path of the extracted data csv file from bags")
parser.add_argument("--image_directory", help="path of the image directory from the extracted data from bags")

args = parser.parse_args()

with open(args.tf_yaml_file, 'r') as tf_yaml_file:
    transform =yaml.load(tf_yaml_file, Loader=yaml.SafeLoader)    

with open(args.camera_info_yaml_file, 'r') as camera_info_yaml_file:
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


tf_base_link_to_opt_frame = Transform()

tf_base_link_to_opt_frame.translation.x = transform['translation']['x']
tf_base_link_to_opt_frame.translation.y = transform['translation']['y']
tf_base_link_to_opt_frame.translation.z = transform['translation']['z']
tf_base_link_to_opt_frame.rotation.x = transform['rotation']['x']
tf_base_link_to_opt_frame.rotation.y = transform['rotation']['y']
tf_base_link_to_opt_frame.rotation.z = transform['rotation']['z']
tf_base_link_to_opt_frame.rotation.w = transform['rotation']['w']


tf_base_link_to_opt_frame_matrix = ros_numpy.numpify(tf_base_link_to_opt_frame)

dtypes = [('image_name', np.object_),
          ('camera_position_x', np.float_),
          ('camera_position_y', np.float_),
          ('cmaera_position_z', np.float_),
          ('camera_orientation_x', np.float_),
          ('camera_orientation_y', np.float_),
          ('camera_orientation_z', np.float_),
          ('camera_orientation_w', np.float_),
          ('gnss_latitude', np.float_),
          ('gnss_longitude', np.float_),
          ('gnss_altitude', np.float_),
          ('gnss_utm_easting', np.float_),
          ('gnss_utm_northing', np.float_),
          ('gnss_velocity_x', np.float_),
          ('gnss_velocity_y', np.float_),
          ('gnss_velocity_z', np.float_)]

load_csv = np.loadtxt(args.csv_file, delimiter=',', dtype=dtypes, skiprows=1)

total_rows = len(load_csv)
chunk_size = 100

for start in range(0, total_rows-chunk_size):
    end = start + chunk_size
    chunk = load_csv[start:end]

    init_pose = Pose()

    init_pose.position.x = chunk[0][1]
    init_pose.position.y = chunk[0][2]
    init_pose.position.z = chunk[0][3]

    init_pose.orientation.x = chunk[0][4]
    init_pose.orientation.y = chunk[0][5]
    init_pose.orientation.z = chunk[0][6]
    init_pose.orientation.w = chunk[0][7]

    # Relative transform and inverse transform
    relative_transform_matrix = ros_numpy.numpify(init_pose)
    inverse_relative_transfrom_matrix = np.linalg.pinv(relative_transform_matrix)

    # future_pose = Pose()
    # future_pose = np.zeros(4)

    future_points_matrix = np.zeros(shape=(chunk.shape[0], 4))
    for id, data in enumerate(chunk):
        
        future_points_matrix[id] = data[1], data[2], data[3], 1
    
    
    relative_points = future_points_matrix.dot(inverse_relative_transfrom_matrix.T)
    relative_points_in_camera_frame = np.dot(tf_base_link_to_opt_frame_matrix, relative_points.T).T

    camera_pixel_coords = np.zeros(shape=(relative_points_in_camera_frame.shape[0], 2))
    z = np.zeros(relative_points_in_camera_frame.shape[0])

    for id, point in enumerate(relative_points_in_camera_frame):
        point3d = point[:3]
        z[id] = point3d[-1]
        camera_pixel_coords[id] = camera_model.project3dToPixel(point3d)
    
    
    image_name = chunk[0][0]

    image_filepath = os.path.join(args.image_directory, image_name)

    image = cv2.imread(image_filepath)
    rectified_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    camera_model.rectifyImage(raw=image, rectified=rectified_image)

    for id, pixel in enumerate(camera_pixel_coords):
        center = (int(pixel[0]), int(pixel[1]))
        scaled_radius = int(np.abs((1 / (z[id]+0.001) * 5)))
        cv2.circle(rectified_image, center, scaled_radius, (255, 0, 0), 5)

    cv2.imshow("Rectified Image", rectified_image)
    cv2.waitKey(50)

cv2.destroyAllWindows()