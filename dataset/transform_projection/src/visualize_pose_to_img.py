import os
import csv
import rospy
import cv2
import numpy as np
import pandas as pd

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import tf2_ros
import tf2_geometry_msgs
import image_geometry
from matplotlib import pyplot as plt

import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from message_filters import ApproximateTimeSynchronizer, Subscriber
from collections import deque

rospy.init_node("visualize_pose_to_img")

message_queue = deque(maxlen=100)
fixed_frame = "map"

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

bridge = CvBridge()
camera_model = PinholeCameraModel()

pose_topic = '/zed2i/zed_node/pose'
camera_info_topic = '/zed2i/zed_node/left/camera_info'
rect_image_topic = '/zed2i/zed_node/left/image_rect_color/compressed'

camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)

camera_model.fromCameraInfo(camera_info_msg)

# rect_images_destination_folder = '/home/adl/catkin_ws/data/images/rect_images'
# rendered_images_destination_folder = '/home/adl/catkin_ws/data/images/rendered_images'
# csv_destination_folder = '/home/adl/catkin_ws/data/csv'


def filter_callback(image_msg, pose_msg):
    # print("Received image and pose messages from the same timestamp")
    # print(f"Image timestamp: {image_msg.header.stamp}")
    # print(f"Image timestamp: {path_msg.header.stamp}")

    message_queue.append((image_msg, pose_msg))
    first_image, first_pose = message_queue[0]
    cv_image = bridge.compressed_imgmsg_to_cv2(first_image,
                                                   desired_encoding='bgr8')

    u = np.zeros(100)
    v = np.zeros(100)
    z = np.zeros(100)

    if len(message_queue) == 100:
        
        for id, (_, pose) in enumerate(message_queue):

            try:
                transformation = tf_buffer.lookup_transform_full(
                    target_frame=first_image.header.frame_id,
                    target_time=first_image.header.stamp,

                    source_frame=pose.header.frame_id,
                    source_time=pose.header.stamp,

                    fixed_frame=fixed_frame,
                    timeout=rospy.Duration(0.0)
                    )
            
                print(f"Transformation: \n {transformation} \n")

                transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transformation)

                print(f"Pose in the past: \n {first_pose} \n")
                print(f"Pose transformed: \n {transformed_pose} \n")

                point_3d = np.array([transformed_pose.pose.position.x,
                                    transformed_pose.pose.position.y,
                                    transformed_pose.pose.position.z])
                
                point_2d = camera_model.project3dToPixel(point_3d)

                print(point_3d)
                print(point_2d)

                u[id] = point_2d[0]
                v[id] = point_2d[1]
                z[id] = point_3d[2] # z -> distance from the camera

                print("---------------------------------------------------------")
            
            except (tf2_ros.ExtrapolationException):
                pass

        for i in range(u.shape[0]):

            scaled_radius = int(np.abs((1 / (z[i]+0.0001) * 5)))

            center = (int(u[i]), int(v[i]))
       
            cv2.circle(cv_image, center, scaled_radius, (255,0,0), 3)        

        cv2.imshow("Rectified Image", cv_image)
        cv2.waitKey(2)
        

# camera_info_sub = Subscriber("/zed/zed_node/left/camera_info", CameraInfo)
image_sub = Subscriber(rect_image_topic, CompressedImage)
pose_sub = Subscriber(pose_topic, PoseStamped)
# path_sub = Subscriber("/zed/zed_node/path_odom", Path)


ats = ApproximateTimeSynchronizer([image_sub, pose_sub], queue_size=10, slop=0.1)
ats.registerCallback(filter_callback)

rospy.spin()