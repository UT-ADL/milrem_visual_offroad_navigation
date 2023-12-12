import argparse
import os

import cv2
import numpy as np
import torch
import yaml

from torchvision.transforms import transforms
import torchvision.transforms.functional as TF

import onnx
import onnxruntime as ort
import time

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist

from PIL import Image

from cv_bridge import CvBridge
from collections import deque

from waypoints_sampling.model.util import load_model
from waypoints_sampling.viz.util_camera import to_camera_frame, load_camera_model


BLUE = (255, 0, 0)
RED  = (0, 0, 255)
GREEN = (0, 255, 0)
FONT_SIZE = 0.3


def parse_arguments():

    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--image_topic',
        default='/zed2i/zed_node/left_raw/image_raw_color/compressed'
    )
    argparser.add_argument(
        '--model_type',
        choices=['vae', 'gnm', 'gnm-pretrained', 'mdn'],
        required=True
    )
    argparser.add_argument(
        '--forward_model_path',
        required=True
    )
    argparser.add_argument(
        '--goal_image_path',
        required=True
    )
    argparser.add_argument(
        '--context',        
        default=True
    )
    argparser.add_argument(
        '--fps',
        type=int,        
        default=15
    )
    
    return argparser.parse_args()


class Visualizer:

    def __init__(self, image_topic, forward_model_path, model_type, goal_image_path, context, fps):
    # def __init__(self, image_topic, forward_model_path, decode_model_path, model_type, waypoint_image_path):
        
        self.TOP_DOWN_VIEW_SCALE = 15

        print(ort.get_device())
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(self.device)

        self.camera_model = load_camera_model()
        self.bridge = CvBridge()
        
        # self.model_path = model_path
        self.forward_model_path = forward_model_path
        # self.decode_model_path = decode_model_path
        
        self.model_type = model_type

        self.config_file = 'src/waypoints_sampling/src/config/' + self.model_type + '.yaml'

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as vae_yaml_file:
                self.model_type_params =yaml.load(vae_yaml_file, Loader=yaml.SafeLoader) 
        else:
            raise Exception(f"Unknown model type {self.model_type}")
        

        self.img_size = self.model_type_params['data_params']['image_size']
        
        # self.img_size.reverse()
        self.context_length = self.model_type_params['model_params']['context_length']
        # self.waypoint_spacing = self.model_type_params['model_params']['waypoint_spacing']
        
        self.fps = fps
        print(f"fps: {self.fps, type(self.fps)}")

        self.i = 0
        self.context = context
        self.obs_img = None
        
        # self.waypoint_spacing = self.fps
        self.waypoint_spacing = 4

        # Deque to store current image along with 75 previous images, 5 images with a spacing of 15 frames in between as context
        # context_length and waypoint spacing can be configured by making changes in the config/vae.yaml file
        # Eventually retrive 6 frames, convert them to tensors and combine to form the obs_img
        
        # self.buffer_length = (self.context_length-1) * self.waypoint_spacing + 1
        
        # if fps = 15, buffer should be 5 * 4 + 1 = 21
        # obs_img --> 0, 4, 8, 12, 16, 20
        if self.fps == 15:
            self.buffer_length = self.context_length * self.waypoint_spacing + 1
        #if fps = 4, buffer--> 6 consecutive images
        elif self.fps == 4:
            self.buffer_length = self.context_length + 1

        self.deque_images = deque(maxlen = self.buffer_length)

        rospy.loginfo("Onnxruntime initializing...")
        
        self.forward_model = self.forward_model_path
        
        start = time.time()
        
        self.forward_session = ort.InferenceSession(self.forward_model,         
                                            providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), 
                                                        "CPUExecutionProvider"])
                                            
        end = time.time()
        print(f"ONNX inference session time: {end-start}")
        rospy.loginfo("Onnxruntime initialized !!")

        self.img_publisher = rospy.Publisher('images_preprocessed',
                                             RosImage,
                                             queue_size=1)
        print("Pre-processed Image Publisher created")

        self.driving_commands_publisher = rospy.Publisher('cmd_vel',
                                                          Twist,
                                                          queue_size=1)
        print("Driving commands Publisher created")
        self.driving_commands = Twist()
        
        self.goal_img_file_path = goal_image_path
        self.goal_img = cv2.imread(self.goal_img_file_path)

        self.goal_img_preprocessed = self.img_tensor_unsqueezed(self.goal_img, self.img_size)
        
        print(f"goal_imge_preprocessed shape: {self.goal_img_preprocessed.shape}")

        self.image_topic = image_topic
        
        

        

        

        # When the incoming image frame rate = 15 Hz
        if self.fps == 15:
            rospy.Subscriber(self.image_topic,
                            CompressedImage,
                            self.image_callback,
                            queue_size=1,
                            buff_size=2*24)
        elif self.fps == 4:
            rospy.Subscriber(self.image_topic,
                            CompressedImage,
                            self.image_callback,
                            queue_size=1,
                            buff_size=2*12)
        else:
            rospy.loginfo("Please enter a valid fps value, i.e., 15 or 4 fps")
            rospy.signal_shutdown("Invalid fps")
        
    def draw_info_overlay(self, frame, goal_x, goal_y, v, w):

        self.draw_rectangle(frame, 5, 15, 200, 120)

        cv2.putText(frame, 'Driving Commands:', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, '    Linear_X: {}'.format(v), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, '    Angular_Z: {}'.format(w), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Goal_x: {}'.format(goal_x), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Goal_y: {}'.format(goal_y), (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        

    # def draw_top_town_overlay(self, frame, waypoint_samples, predicted_wps):
    def draw_top_town_overlay(self, frame, predicted_wps):
        self.draw_rectangle(frame, 10, 210, 140, 130)

        current_position = (75, 325)

        # waypoint_samples = waypoint_samples.reshape(20, 5, 2)

        # if waypoint_samples is not None:
        #     for i in range(waypoint_samples.shape[0]):
        #         wp_trajectory = waypoint_samples[i]
        #         self.draw_top_down_trajectory(frame, wp_trajectory, current_position, BLUE)

        predicted_wps = predicted_wps.reshape(-1, 2)
        self.draw_top_down_trajectory(frame, predicted_wps, current_position, RED)
        cv2.circle(frame, current_position, 2, (0, 255, 255), 2)
        

    def draw_top_down_trajectory(self, frame, predicted_wps, current_position, color):

        for i in range(predicted_wps.shape[0]):
            pred_wp = predicted_wps[i]
            
            scaled_current_wp = (current_position[0] - int(self.TOP_DOWN_VIEW_SCALE * pred_wp[1]),
                                 current_position[1] - int(self.TOP_DOWN_VIEW_SCALE * pred_wp[0]))
            cv2.circle(frame, scaled_current_wp, 2, color, 2)

            if i > 0:
                pred_prev_wp = predicted_wps[i - 1]
                scaled_prev_wp = (current_position[0] - int(self.TOP_DOWN_VIEW_SCALE * pred_prev_wp[1]),
                                  current_position[1] - int(self.TOP_DOWN_VIEW_SCALE * pred_prev_wp[0]))
                cv2.line(frame, scaled_prev_wp, scaled_current_wp, color, 1)
    

    def draw_waypoint_img(self, frame, goal_img):
        print(f"Goal img shape: {goal_img.shape}")
        scale_percent = 15  # percent of original size
        width = int(goal_img.shape[1] * scale_percent / 100)
        height = int(goal_img.shape[0] * scale_percent / 100)        
        dim = (width, height)
        print(f"dim: {dim}")

        # resize image
        resized = cv2.resize(goal_img, dim, interpolation=cv2.INTER_AREA)

        spacing = 10
        frame[spacing:resized.shape[0]+spacing, frame.shape[1]-resized.shape[1]-spacing:frame.shape[1]-spacing] = resized


    def draw_rectangle(self, frame, x, y, w, h):
        sub_img = frame[y:y+h, x:x+w]
        rect = np.full(sub_img.shape, (0, 0, 0), np.uint8)
        alpha = 0.4
        res = cv2.addWeighted(sub_img, alpha, rect, 1-alpha, 0)
        frame[y:y+h, x:x+w] = res


    def show_next_frame(self, img):
        img_resized = cv2.resize(img, (2*img.shape[1], 2*img.shape[0]), interpolation=(cv2.INTER_AREA))
        cv2.imshow('Live camera frame', img_resized)
    
    def rectify_image(self, img):
        rectified_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.camera_model.rectifyImage(raw=img, rectified=rectified_image)
        return rectified_image
    
    def image_callback(self, img_msg):
        
        img_processing_init_time = time.time()

        img = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        img_preprocessed = self.img_tensor_unsqueezed(img, self.img_size)

        print(f"Shape of image after preprocessing: {img_preprocessed.shape}")

        img_processing_finish_time = time.time()

        print(f"It took {img_processing_finish_time - img_processing_init_time} seconds for processing a single image")
        
        

        if self.context == True:
            
            self.deque_images.append(img_preprocessed)

            if len(self.deque_images) == self.buffer_length:
                
                # images arriving in 15 Hz freq
                if self.fps == 15:

                    buffer_init_time = time.time()
                    deque_tensors = deque(maxlen = (self.context_length + 1))

                    for i in range(0, self.buffer_length, self.waypoint_spacing):
                        deque_tensors.append(self.deque_images[i])
                    
                    self.obs_img = np.concatenate(tuple(deque_tensors), axis=1)
                    buffer_finish_time = time.time()

                    print(f"It took {buffer_finish_time - buffer_init_time} seconds to create image buffer and concat image batches")
                
                # images arriving in 4 Hz freq
                elif self.fps == 4:
                    
                    self.obs_img = np.concatenate(tuple(self.deque_images), axis=1)
                
                
        # no-context
        else:
            self.obs_img = img_preprocessed

        if self.obs_img is not None:

            print(f"obs_img shape: {self.obs_img.shape}")

            start_time = time.time()  

            forward_model_in = {'obs_img': self.obs_img,
                                'goal_img': self.goal_img_preprocessed}
            forward_model_out = self.forward_session.run(None, forward_model_in)

            end_time = time.time()

            # print("Model prediction")
            
            print(f"It took {end_time - start_time} seconds for inference")
            # print("--------------------------")
            # self.i += 1
            # print(self.i)

            # forward_model_out = np.array(forward_model_out)
            # print(np.squeeze(forward_model_out[1]), type(forward_model_out[1]))
            # print(forward_model_out[1][:, 3])
            
            predicted_actions = np.squeeze(forward_model_out[1])
            print(predicted_actions[:, 0:2])
            # waypoint_samples = np.squeeze(forward_model_out[3])

            # if self.context == True:

                # predicted_actions = np.cumsum(predicted_actions, axis=0)
                # waypoint_samples = np.cumsum(waypoint_samples, axis=1)

            # waypoint_samples = waypoint_samples.reshape(-1,2)

            goal_x, goal_y = predicted_actions[-1][:2]
            dist = np.sqrt((goal_x**2) + (goal_y**2))
            theta = np.arctan2(goal_y, goal_x)

            if dist <= 1.0:
                v = 0.0
                print("Goal reached")
                rospy.signal_shutdown("Goal reached")
            else:
                v = dist

            w = theta
            
            print(f"v, w: {v, w}")

            v = np.clip(v, 0, 0.25)
            w = np.clip(w, -0.6, 0.6)
            
            self.driving_commands.linear.x = v
            self.driving_commands.angular.z = w
            
            visualization_init_time = time.time()

            rectified_img = self.rectify_image(img)
            to_camera_frame(rectified_img, predicted_actions[:, 0:2], RED)
            self.draw_waypoint_img(rectified_img, self.goal_img)
            self.draw_info_overlay(rectified_img, goal_x, goal_y, v, w)
            self.draw_top_town_overlay(rectified_img, predicted_actions[:, 0:2])

            self.show_next_frame(rectified_img)
            cv2.waitKey(1)

            visualization_finish_time = time.time()

            print(f"It took {visualization_finish_time - visualization_init_time} seconds for visualization")

            self.driving_commands_publisher.publish(self.driving_commands)
            print(f"Publihsed cmd_vel --> linear.x, angular.z: {v, w}")

            print("---------------------------------------")
            print("---------------------------------------")

    
    def img_tensor_unsqueezed(self, img, crop_size):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # print(f"Img shape: {img.size}")
        
        w, h = img.size
        # print(f"Crop size: {crop_size}")
        # w, h, c = img.shape
        
        aspect_ratio = crop_size[0] / crop_size[1]
        # print(f"Aspect ratio: {aspect_ratio}")

        img = TF.center_crop(img, [h, int(h * aspect_ratio)])
        img = TF.resize(img, [crop_size[1], crop_size[0]], antialias=True)

        img_msg = self.bridge.cv2_to_imgmsg(np.array(img), encoding='rgb8')
        img_msg.header.stamp = rospy.Time.now()
        
        self.img_publisher.publish(img_msg)
        
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(dim=0)

        return img_tensor.numpy()


if __name__ == "__main__":
    rospy.init_node("Inference_Visualization")

    args = parse_arguments()

    image_topic = args.image_topic
    forward_model_path = args.forward_model_path
    model_type = args.model_type
    goal_image_path = args.goal_image_path
    context = args.context
    fps = args.fps

    node = Visualizer(image_topic = image_topic,
                      forward_model_path = forward_model_path,
                      model_type = model_type,
                      goal_image_path = goal_image_path,
                      context = context,
                      fps = fps)
    rospy.spin()