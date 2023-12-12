#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import Image, NavSatFix
import imghdr
import exifread
import cv2
import numpy as np
from cv_bridge import CvBridge

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def extract_gps_info(image_stream):
    tags = exifread.process_file(image_stream)
    try:
        lat = [float(x.num) / float(x.den) for x in tags['GPS GPSLatitude'].values]
        lon = [float(x.num) / float(x.den) for x in tags['GPS GPSLongitude'].values]
    except KeyError:
        return None
    return lat, lon

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: image_publisher_node.py <image_file>')
        sys.exit(1)

    # Initialize ROS node and publishers
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('goal_image', Image, queue_size=10, latch=True)
    gps_pub = rospy.Publisher('goal_gps', NavSatFix, queue_size=10, latch=True)
    bridge = CvBridge()

    # Convert the uploaded image to a ROS Image using cv_bridge
    image_file = sys.argv[1]
    image_type = imghdr.what(image_file)
    if image_type not in ALLOWED_EXTENSIONS:
        print('File type not supported. Please upload a JPEG image.')
        sys.exit(1)

    lat_lon = extract_gps_info(open(image_file, 'rb'))
    if not lat_lon:
        print('Image does not contain GPS information.')
        sys.exit(1)

    # Convert the uploaded image to a ROS Image using cv_bridge
    cv_image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    image_msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    image_pub.publish(image_msg)

    # Publish the GPS data
    lat, lon = lat_lon
    gps_msg = NavSatFix()
    gps_msg.header.stamp = rospy.Time.now()
    gps_msg.latitude = sum(x / 60**n for n, x in enumerate(lat))
    gps_msg.longitude = sum(x / 60**n for n, x in enumerate(lon))
    gps_pub.publish(gps_msg)

    # Stay alive to publish latched messages
    rospy.spin()
