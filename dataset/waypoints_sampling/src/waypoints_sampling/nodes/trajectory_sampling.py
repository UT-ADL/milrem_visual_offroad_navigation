import argparse
import os

import cv2
import numpy as np
import torch
import yaml

from torchvision.transforms import transforms

from waypoints_sampling.model.util import load_model
from waypoints_sampling.viz.util_camera import to_camera_frame, load_camera_model

import rospy
from sensor_msgs.msg import CompressedImage

from cv_bridge import CvBridge
import cv2

from collections import deque
import time

BLUE = (255, 0, 0)
# GREEN = (0, 255, 0)
RED = (0, 0, 255)
# FONT_SIZE = 0.3

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
        '--model_path',
        required=True
    )
    argparser.add_argument(
        '--waypoint_image_path',
        required=True
    )
    return argparser.parse_args()

class Visualizer:

    def __init__(self, image_topic, model_path, model_type, waypoint_image_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.camera_model = load_camera_model()
        self.bridge = CvBridge()
        
        self.waypoint_img_file_path = waypoint_image_path
        self.wp_img = cv2.imread(self.waypoint_img_file_path)

        self.model_path = model_path
        self.model_type = model_type

        self.config_file = 'src/waypoints_sampling/src/config/' + self.model_type + '.yaml'

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as vae_yaml_file:
                self.model_type_params =yaml.load(vae_yaml_file, Loader=yaml.SafeLoader) 
        else:
            raise Exception(f"Unknown model type {self.model_type}")

        self.img_size = self.model_type_params['data_params']['image_size']
        self.img_size.reverse()

        self.context_length = self.model_type_params['model_params']['context_length']
        self.waypoint_spacing = self.model_type_params['model_params']['waypoint_spacing']

        self.buffer_length = self.context_length * self.waypoint_spacing + 1
        self.deque_images = deque(maxlen = self.buffer_length)

        self.waypoint_img_file_path = waypoint_image_path
        self.wp_img = cv2.imread(self.waypoint_img_file_path)

        self.i = 0
        self.img_dir = "test_img_dir"
        os.makedirs(self.img_dir, exist_ok=True)

        with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
            
        self.model = load_model(model_path=self.model_path,
                        model_type=self.model_type,
                        model_config=config)
        self.model.to(device=self.device)
        
        self.image_topic = image_topic

        self.bridge = CvBridge()

        rospy.Subscriber(self.image_topic,
                         CompressedImage,
                         self.image_callback)
    

    def draw_trajectory(self, frame , waypoint_samples, predicted_wps):

        self.draw_rectangle(frame, 10, 210, 140, 130)

        x_position = 325
        y_position = 75

        cv2.circle(frame, (y_position, x_position), 2, (0, 255, 255), 3)

        scale = 15

        if waypoint_samples is not None:
        
            for i in range(waypoint_samples.shape[0]):
                x, y = waypoint_samples[i]
                cv2.circle(frame, (y_position-int(scale*y), x_position-int(scale*x)), 1, BLUE, 1)
        
        for i in range(predicted_wps.shape[0]):
            pred_wp = predicted_wps[i]
            cv2.circle(frame, (y_position - int(scale * pred_wp[1]), x_position - int(scale * pred_wp[0])), 2, RED, 2)


    def draw_rectangle(self, frame, x, y, w, h):
        sub_img = frame[y:y+h, x:x+w]
        rect = np.full(sub_img.shape, (0, 0, 0), np.uint8)
        alpha = 0.4
        res = cv2.addWeighted(sub_img, alpha, rect, 1-alpha, 0)
        frame[y:y+h, x:x+w] = res


    def show_next_frame(self, obs_img):
        obs_img_resized = cv2.resize(obs_img, (2*640, 2*360), interpolation=(cv2.INTER_AREA))
        cv2.imshow('Live camera frame', obs_img_resized)
    
    def rectify_image(self, obs_img):
        obs_img = cv2.cvtColor(np.array(obs_img), cv2.COLOR_RGB2BGR)    
        rectified_image = np.zeros((obs_img.shape[0], obs_img.shape[1], 3), dtype=np.uint8)
        self.camera_model.rectifyImage(raw=obs_img, rectified=rectified_image)
        return rectified_image
    
    def image_callback(self, img_msg):
        # print(self.device)
        self.i += 1
        print(self.i)
        img = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        
        self.deque_images.append(img)

        if len(self.deque_images) == self.buffer_length:
            start_time = time.time()
            deque_tensors = deque(maxlen=(self.context_length+1))
            

            for i in range(0, self.buffer_length, self.waypoint_spacing):
                img_tensor = self.img_tensor_unsqueezed(self.deque_images[i], self.img_size)
                deque_tensors.append(img_tensor)
            
            obs_img_tensor = torch.cat(tuple(deque_tensors), dim=1)
            
            print(obs_img_tensor.dtype, obs_img_tensor.device, obs_img_tensor.shape)
            goal_img_tensor = self.img_tensor_unsqueezed(self.wp_img, self.img_size)
            print(goal_img_tensor.dtype, goal_img_tensor.device, goal_img_tensor.shape)

            with torch.no_grad():
                
                predictions, _, _, _ = self.model(obs_img_tensor, goal_img_tensor)
                distance, actions = self.model.sample(obs_img_tensor, 20)
            
            predicted_actions = predictions[1].squeeze().detach().cpu().numpy() 

            waypoint_samples = actions.detach().cpu().numpy() 
            waypoint_samples = waypoint_samples.reshape(-1, 2)

            end_time = time.time()

            print(f"time taken by model: {end_time-start_time}")

            rectified_img = self.rectify_image(self.deque_images[-1])
        
            to_camera_frame(rectified_img, waypoint_samples, BLUE)
            to_camera_frame(rectified_img, predicted_actions, RED)

            self.draw_trajectory(rectified_img, waypoint_samples, predicted_actions)
            self.show_next_frame(rectified_img)  
            
            print("-------------------------------------------")         

            cv2.waitKey(1)

    def img_tensor_unsqueezed(self, img, img_size):
        transform = transforms.Compose(
            [                
                transforms.ToTensor(),
                transforms.Resize(size=img_size, antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ]
        )
        img_tensor = transform(img).to(device=self.device, dtype=torch.float32)
        return img_tensor.unsqueeze(dim=0)


if __name__ == "__main__":
    rospy.init_node('viz_predictions')

    args = parse_arguments()
    image_topic = args.image_topic
    model_path = args.model_path
    model_type = args.model_type
    waypoint_image_path = args.waypoint_image_path

    node = Visualizer(image_topic, model_path, model_type, waypoint_image_path)    

    rospy.spin()