import os
import csv
import rospy
import cv2
import numpy as np
import pandas as pd

from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path, Odometry

from message_filters import ApproximateTimeSynchronizer, Subscriber

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

import tf2_ros
import tf

import matplotlib.pyplot as plt

np.float = np.float64
import ros_numpy


if __name__ == "__main__":
    try:
        rospy.init_node("camera_pose_to_optical_frame")

        tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(tf_buffer)

        target_frame = 'zed2i_left_camera_optical_frame'
        source_frame = 'base_link'

        rate = rospy.Rate(10)
        
        transformation = None
        translation = None
        rotation = None

        while not rospy.is_shutdown():
            try:
                transformation = tf_buffer.lookup_transform(target_frame=target_frame,
                                                            source_frame=source_frame,
                                                            time=rospy.Time())
                break   # Once the trnasformation between the 2 frames is received, break the lookup loop

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform not available yet, waiting...")
                rate.sleep()
        
        rospy.loginfo("Transform received: \n {}".format(transformation))

        translation = [transformation.transform.translation.x,
                       transformation.transform.translation.y,
                       transformation.transform.translation.z]
        
        rotation = [transformation.transform.rotation.x,
                    transformation.transform.rotation.y,
                    transformation.transform.rotation.z,
                    transformation.transform.rotation.w]
        
        # tf_matrix_from_base_link_to_left_camer_optical_frame
        base_link_to_optical_frame_tf_matrix = tf.transformations.compose_matrix(translate=translation,
                                                                 angles=tf.transformations.euler_from_quaternion(rotation))
        
        rospy.loginfo("Transformation Matrix: \n {}".format(base_link_to_optical_frame_tf_matrix))

        image_directory = "/home/anish/catkin_ws/out/images/"
        csv_file = "/home/anish/catkin_ws/out/csv/extracted_data.csv"

        bridge = CvBridge()
        camera_model = PinholeCameraModel()

        camera_info_topic = '/zed2i/zed_node/left_raw/camera_info'

        camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
        camera_model.fromCameraInfo(camera_info_msg)

        camera_matrix = np.reshape(camera_info_msg.K, (3,3))
        distortion_coeffs = np.array(camera_info_msg.D)

        # Take 100 first rows from csv file
        data_df = pd.read_csv(csv_file)
        total_rows = len(data_df)
        chunk_size = 100

        for start in range(0, total_rows-100):
            end = start + chunk_size
            chunk = data_df[start:end]

            data_matrix = chunk.values      # 2-dim np array extracted from the 100 rows of the csv file  
            print(data_matrix)

            init_pose = Pose()

            init_pose.position.x = data_matrix[0][1]        # camera_position_x
            init_pose.position.y = data_matrix[0][2]        # camera_position_y
            init_pose.position.z = data_matrix[0][3]        # camera_position_z

            init_pose.orientation.x = data_matrix[0][4]     # camera_orientation_x
            init_pose.orientation.y = data_matrix[0][5]     # camera_orientation_y
            init_pose.orientation.z = data_matrix[0][6]     # camera_orientation_z
            init_pose.orientation.w = data_matrix[0][7]     # camera_orientation_w

            # Relative transform and inverse transform
            transform_matrix = ros_numpy.numpify(init_pose)
            inverse_transform_matrix = np.linalg.pinv(transform_matrix)

            future_pose = Pose()

            future_points_matrix = np.zeros(shape=(data_matrix.shape[0], 4))
            for id, data in enumerate(data_matrix):
                future_pose.position.x, future_pose.position.y, future_pose.position.z  = data[1], data[2], data[3]
                future_pose.orientation.x, future_pose.orientation.y, future_pose.orientation.z, future_pose.orientation.w = data[4], data[5], data[6], data[7]

                future_point = ros_numpy.numpify(future_pose.position, hom=True)
                future_points_matrix[id] = future_point
            
            relative_points = future_points_matrix.dot(inverse_transform_matrix.T)
            relative_points_in_camera_frame = np.dot(base_link_to_optical_frame_tf_matrix, relative_points.T).T

            rospy.loginfo("Relative coords in camera optical frame: \n {} \n".format(relative_points_in_camera_frame))

            camera_pixel_coords = np.zeros(shape=(relative_points_in_camera_frame.shape[0], 2))
            for i in range(relative_points_in_camera_frame.shape[0]):
                point_3d = np.array([relative_points_in_camera_frame[i][0],
                                     relative_points_in_camera_frame[i][1],
                                     relative_points_in_camera_frame[i][2]
                                     ])
                camera_pixel_coords[i] = camera_model.project3dToPixel(point_3d)
            
            image_name = data_matrix[0][0]
            
            image_file_path = os.path.join(image_directory, image_name)

            image = cv2.imread(image_file_path)
            rectified_image = cv2.undistort(image, camera_matrix, distortion_coeffs)

            radius = 2
            for i in range(camera_pixel_coords.shape[0]):
                center = (int(camera_pixel_coords[i][0]), int(camera_pixel_coords[i][1]))
                cv2.circle(rectified_image, center, radius, (255, 0, 0), 5)

            cv2.imshow("Rectified Image", rectified_image)
            cv2.waitKey(30)
            
        cv2.destroyAllWindows()

        
    except Exception as e:
        rospy.logwarn(e)
    



    
        



        

        