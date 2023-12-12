import argparse
import json
import os

import cv2
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CompressedImage
from tf.transformations import euler_from_quaternion

rospy.init_node("extract_frames")

parser = argparse.ArgumentParser()

parser.add_argument("--rect_image_topic", type=str, help="Rectified Image topic name", default="/zed2i/zed_node/left/image_rect_color/compressed")
parser.add_argument("--camera_info_topic", type=str, help="Camera Info topic name", default="/zed2i/zed_node/left/camera_info")
parser.add_argument("--pose_topic", type=str, help="Pose topic name", default="/zed2i/zed_node/pose")
parser.add_argument("--location_topic", type=str, help="Location topic name", default="/filter/positionlla")
parser.add_argument("--image_directory", type=str, help="Directory where the images will be stored")
parser.add_argument("__name", type=str, help="Name")
parser.add_argument("__log", type=str, help="Log")
args = parser.parse_args()

rect_image_topic = args.rect_image_topic
camera_info_topic = args.camera_info_topic
pose_topic = args.pose_topic
location_topic = args.location_topic
image_directory = args.image_directory

bridge = CvBridge()

if not os.path.exists(image_directory):
    os.makedirs(image_directory)


def filter_callback(image_msg, pose_msg, location_msg):
    cv_image = bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    image_name = "img" + '{:05d}.jpg'.format(image_msg.header.seq)
    image_filepath = os.path.join(image_directory, image_name)
    cv2.imwrite(image_filepath, cv_image)

    json_filename = "img" + '{:05d}.json'.format(image_msg.header.seq)
    json_filepath = os.path.join(image_directory, json_filename)

    quaternion = [
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
        pose_msg.pose.orientation.z, pose_msg.pose.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(quaternion)

    meta_json = {
        "image_name": "img" + '{:05d}.jpg'.format(pose_msg.header.seq),
        "pos_x": pose_msg.pose.position.x,
        "pos_y": pose_msg.pose.position.y,
        "pos_z": pose_msg.pose.position.z,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "longitude": location_msg.vector.x,
        "latitude": location_msg.vector.y,
        "attitude": location_msg.vector.z

    }
    with open(json_filepath, "w") as json_file:
        json_file.write(json.dumps(meta_json, indent=4))

image_sub = Subscriber(rect_image_topic, CompressedImage)
pose_sub = Subscriber(pose_topic, PoseStamped)
position_sub = Subscriber(location_topic, Vector3Stamped)

ats = ApproximateTimeSynchronizer([image_sub, pose_sub, position_sub], queue_size=10, slop=0.1)
ats.registerCallback(filter_callback)

rospy.spin()