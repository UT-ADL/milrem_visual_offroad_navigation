import os
import rospy
import csv
import cv2
import numpy as np
import pandas as pd
from collections import deque
import argparse

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

import rospy
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
import tf2_geometry_msgs


rospy.init_node("projection_distance")
fixed_frame = "map"

parser = argparse.ArgumentParser()

parser.add_argument("--rect_image_topic", type=str, help="Rectified Image topic name", default="/zed2i/zed_node/left/image_rect_color/compressed")
parser.add_argument("--camera_info_topic", type=str, help="Camera Info topic name", default="/zed2i/zed_node/left/camera_info")
parser.add_argument("--pose_topic", type=str, help="Pose topic name", default="/zed2i/zed_node/pose")
parser.add_argument("--image_directory", type=str, help="Directory where the images will be stored")
parser.add_argument("--csv_directory", type=str, help="Directory where the csv files will be stored")
parser.add_argument("--queue_length", type=int, help="Max length of the deque")
parser.add_argument("__name", type=str, help="Name")
parser.add_argument("__log", type=str, help="Log")

args = parser.parse_args()

rect_image_topic = args.rect_image_topic
camera_info_topic = args.camera_info_topic
pose_topic = args.pose_topic
image_directory = args.image_directory
csv_directory = args.csv_directory
max_len = args.queue_length


message_queue = deque(maxlen=max_len)

tf_buffer = tf2_ros.Buffer(rospy.Duration(100))
tf_listener = tf2_ros.TransformListener(tf_buffer)

bridge = CvBridge()
camera_model = PinholeCameraModel()

camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
camera_model.fromCameraInfo(camera_info_msg)

rect_images_destination_folder = image_directory + "/rect_images"
rendered_images_destination_folder = image_directory + "/rendered_images"
csv_destination_folder = csv_directory

if not os.path.exists(rect_images_destination_folder):
    os.makedirs(rect_images_destination_folder)
if not os.path.exists(rendered_images_destination_folder):
    os.makedirs(rendered_images_destination_folder)
if not os.path.exists(csv_destination_folder):
    os.makedirs(csv_destination_folder)

def compute_distance(point1, point2):

    return (np.linalg.norm(point1 - point2))

def filter_callback(image_msg, pose_msg):
    
    rospy.loginfo(len(message_queue))
    if not message_queue:
        message_queue.append((image_msg, pose_msg))
    
    last_pose = message_queue[-1][-1]
    last_pose_arr = np.array([last_pose.pose.position.x, last_pose.pose.position.y, last_pose.pose.position.z])
    new_pose_arr = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
    
    dist = compute_distance(last_pose_arr, new_pose_arr)

    if dist >= 1.0:
        message_queue.append((image_msg, pose_msg))

    first_image, first_pose = message_queue[0]
    cv_image = bridge.compressed_imgmsg_to_cv2(first_image,
                                               desired_encoding='bgr8')
    rect_img = cv_image

    u = np.zeros(max_len)
    v = np.zeros(max_len)
    z = np.zeros(max_len)

    if len(message_queue) == max_len:

        rect_image_name = "img" + '{:05d}.jpg'.format(message_queue[0][0].header.seq)
        rect_image_filepath = os.path.join(rect_images_destination_folder, rect_image_name)
        cv2.imwrite(rect_image_filepath, rect_img)

        csv_name = "img" + '{:05d}.csv'.format(message_queue[0][0].header.seq)
        csv_filepath = os.path.join(csv_destination_folder, csv_name)
        csv_headers = ["image_name", "x", "y", "z", "u", "v"]

        with open(csv_filepath, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers)
        
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

                    transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transformation)

                    point_3d = np.array([transformed_pose.pose.position.x,
                                        transformed_pose.pose.position.y,
                                        transformed_pose.pose.position.z])
                    
                    point_2d = camera_model.project3dToPixel(point_3d)

                    u[id] = point_2d[0]
                    v[id] = point_2d[1]
                    z[id] = point_3d[2]
                   
                    writer.writerow(["img" + '{:05d}.jpg'.format(message_queue[0][0].header.seq), point_3d[0], point_3d[1], point_3d[2], point_2d[0], point_2d[1]])
                
                except Exception as e:
                    rospy.loginfo(e)
                    

            for i in range(u.shape[0]):
                center = (int(u[i]), int(v[i]))
                scaled_radius = int(np.abs((1 / (z[i]+0.0001) * 5)))
                # cv2.circle(cv_image, center, 2, (255,0,0), 2)
                cv2.circle(cv_image, center, scaled_radius, (255,0,0), 3)
            
            rendered_image_name = rect_image_name
            rendered_image_filepath = os.path.join(rendered_images_destination_folder, rendered_image_name)
            cv2.imwrite(rendered_image_filepath, cv_image)
            
            cv2.imshow("Rectified Image", cv_image)
            cv2.waitKey(1) 
    

image_sub = Subscriber(rect_image_topic, CompressedImage)
pose_sub = Subscriber(pose_topic, PoseStamped)

ats = ApproximateTimeSynchronizer([image_sub, pose_sub], queue_size=10, slop=0.1)
ats.registerCallback(filter_callback)

rospy.spin()