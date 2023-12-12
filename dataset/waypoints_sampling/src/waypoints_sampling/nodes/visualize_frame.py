import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped

import cv2
import numpy as np



class VisualizeFrame:
    def __init__(self):
        self.bridge = CvBridge()

        # subscription image topic
        self.rect_image_topic = '/zed2i/zed_node/left/image_rect_color/compressed'

        # Subscription
        rospy.Subscriber(self.rect_image_topic, 
                         CompressedImage,
                         self.image_callback)



    def publish_img(self, img, stamp):
        pass

    def image_callback(self, img_msg):
        img = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        cv2.imshow("Rectified Iamge", img)
        cv2.waitKey(50)

    # def run(self):
    #     rospy.spin()

if __name__ == "__main__":
    rospy.init_node('viz_frame')
    node = VisualizeFrame()
    rospy.spin()