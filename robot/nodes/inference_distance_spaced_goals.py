import os
import cv2
import numpy as np
import yaml
import time

import onnxruntime as ort

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo, Joy
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from image_geometry import PinholeCameraModel
import tf2_geometry_msgs
import tf2_ros

from utils.viz_util import show_next_frame, rectify_image, visualize_waypoints, draw_rectangle, draw_waypoint_image, draw_top_down_overlay
from utils.preprocessing_images import center_crop_and_resize, prepare_image
from utils.batching import batch_obs_plus_context

from cv_bridge import CvBridge
from collections import deque

# COLORS in BGR format
BLUE = (255, 0, 0)
RED  = (0, 0, 255)
GREEN = (0, 255, 0)
FONT_SIZE = 0.3

class Visualizer:
    
    def __init__(self):

        # Fetch parameters
        self.local_config_dir_path = rospy.get_param('~local_config_directory_path')
        self.extracted_data_directory = rospy.get_param('~extracted_data_directory_path')
        self.local_model_type = rospy.get_param('~local_model_type')
        self.local_model_path = rospy.get_param('~local_model_path')
        self.fps = int(rospy.get_param('~fps'))
        self.goals_spacing = rospy.get_param('~goals_spacing')
        self.odometry_tolerance = rospy.get_param('~odometry_tolerance')
        self.goal_conditioning_threshold = rospy.get_param('~goal_conditioning_threshold')
        self.base_link_frame = rospy.get_param('~base_link_frame', 'base_link')
        self.left_camera_frame = rospy.get_param('~left_camera_frame', 'zed_left_camera_frame')
        self.left_camera_optical_frame = rospy.get_param('~left_camera_optical_frame', 'zed_left_camera_optical_frame')
        self.record_video = rospy.get_param('~record_video')

        # Initialize tf2 listener and CV bridge
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()

        # Load local planner configration
        local_planner_config_file = os.path.join(self.local_config_dir_path, str(self.local_model_type) + '.yaml')
        if os.path.exists(local_planner_config_file):
            with open(local_planner_config_file, 'r') as f:
                self.local_planner_config = yaml.safe_load(f)
        else:
            raise Exception(f"No such file: {local_planner_config_file}")
        rospy.loginfo(f"Loaded local planner config: {local_planner_config_file}")

        # Fetch local planner configuration parameters
        self.img_size = self.local_planner_config['data_params']['image_size']
        self.waypoint_length = self.local_planner_config['model_params']['len_traj_pred']
        self.context_length = self.local_planner_config['model_params']['context_size']      

        self.waypoint_spacing = self.local_planner_config['model_params']['waypoint_spacing']

        rospy.loginfo(f"Observation size: {self.img_size}")
        
        self.local_session = ort.InferenceSession(self.local_model_path, 
                providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
            )        
        rospy.loginfo(f"Loaded local planner model: {self.local_model_path}")

        # Load camera model
        camera_info_msg = rospy.wait_for_message('/zed/zed_node/left_raw/camera_info', CameraInfo)
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(camera_info_msg)
        rospy.loginfo("Loaded pinhole camera model")

        # Prepare context image buffers
        # if fps = 15, buffer should be 5 * 4 + 1 = 21
        # obs_img --> 0, 4, 8, 12, 16, 20
        if self.fps == 15:
            self.buffer_length = self.context_length * self.waypoint_spacing + 1
        #if fps = 4, buffer--> 6 consecutive images
        elif self.fps == 4:
            self.buffer_length = self.context_length + 1
        else:
            raise Exception(f"FPS {self.fps} not supported")

        self.deque_images = deque(maxlen = self.buffer_length)

        # Initialize video writer
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video = cv2.VideoWriter(self.record_video, fourcc, self.fps, (640, 360))
            rospy.loginfo(f"Recording video to: {self.record_video}")

        # Sampling images based on distance
        sampled_img_names = []
        
        csv_dir = os.path.join(self.extracted_data_directory, 'csv')
        image_dir = os.path.join(self.extracted_data_directory, 'images')

        dtypes = [('image_name', 'U20'), ('timestamp', 'f8'), ('position_x', 'f8'), ('position_y', 'f8')]
        extracted_data = np.genfromtxt(os.path.join(csv_dir, 'extracted_data.csv'), delimiter=',', skip_header=True, dtype=dtypes)

        init_x, init_y = extracted_data[0][2], extracted_data[0][3]

        for data in extracted_data:
            init_pose = np.array([init_x, init_y])
            final_x, final_y = data[2], data[3]
            final_pose = np.array([final_x, final_y])

            if self.calculate_distance(init_pose, final_pose) >= self.goals_spacing:
                sampled_img_names.append(str(data[0]))
                init_x = final_x
                init_y = final_y
        sampled_img_names.append(extracted_data[-1][0])

        # For storing distance-spaced goal images
        self.sampled_img_batch = None

        # for img_name in img_names:
        for img_name in sampled_img_names:
            img = cv2.imread(os.path.join(image_dir, img_name))

            if self.sampled_img_batch is None:
                self.sampled_img_batch = img[np.newaxis, :]
            else:
                self.sampled_img_batch = np.concatenate((self.sampled_img_batch, img[np.newaxis, :]), axis=0)
        #----------------------------------------------------

        # Initialize internal variables
        self.obs_img = None
        self.goal_img = None 

        self.goal_img_index = 0

        self.successful_goal_completions = 0
        self.num_interventions = 0

        self.goal_num = None
        self.completion_rate = None
        self.final_goal_passed = False 

        self.prev_pose = None
        self.mode = "automatic"
        
        self.odom_topic = '/odometry/filtered'
        self.joy_topic = '/bluetooth_teleop/joy'
        self.image_topic = '/zed/zed_node/left_raw/image_raw_color/compressed'

        self.goal_in_left_camera_frame = PointStamped()        
        self.goal_in_base_link = PointStamped()
        self.driving_command = Twist()


        # Fetch transforms
        self.tf_from_cam_frame_to_base_link = self.tf_buffer.lookup_transform(
                                target_frame=self.base_link_frame,
                                source_frame=self.left_camera_frame,
                                time=rospy.Time.now(),
                                timeout=rospy.Duration(10)
                        )
        self.tf_from_cam_frame_to_optical_frame = self.tf_buffer.lookup_transform(
                                target_frame=self.left_camera_optical_frame,
                                source_frame=self.left_camera_frame,
                                time=rospy.Time.now(),
                                timeout=rospy.Duration(10)
                        )

        # Initialize ROS publishers and subscribers
        self.driving_command_publisher = rospy.Publisher('cmd_vel',
                                                          Twist,
                                                          queue_size=1)

        rospy.Subscriber(self.odom_topic,
                         Odometry,
                         self.odom_callback,
                         queue_size=1)        
        
        rospy.Subscriber(self.joy_topic,
                         Joy,
                         self.joy_callback,
                         queue_size=1)
        
        rospy.Subscriber(self.image_topic,
                        CompressedImage,
                        self.image_callback,
                        queue_size=1,
                        buff_size=2**24)

        
    def calculate_distance(self, init_pose, final_pose):    
        return np.linalg.norm(np.array(init_pose) - np.array(final_pose))
    
    def draw_info_overlay(self, frame, predicted_distance, goal_x, goal_y, v, w):

        # self.draw_rectangle(frame, 5, 2, 160, 250)
        draw_rectangle(frame, 5, 2, 160, 240)
        total_goals = self.sampled_img_batch.shape[0]
        op = "/"

        if self.final_goal_passed == True:
            total_goals_passed = self.goal_num
            self.completion_rate = self.successful_goal_completions / total_goals_passed
            goal_num_text = " "
            total_goals = " "
            op = " "
            
        else:
            if self.goal_img_index == 0:
                total_goals_passed = 0
                self.completion_rate = 0
            else:
                total_goals_passed = self.goal_num - 1
                if total_goals_passed > 0:
                    self.completion_rate = self.successful_goal_completions / total_goals_passed
                else:
                    self.completion_rate = 0

            goal_num_text = self.goal_num       

        cv2.putText(frame, 'Driving Commands:', (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, '  Linear_X: {:.2f}'.format(v), (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, '  Angular_Z: {:.2f}'.format(w), (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, 'Goal_x: {:.2f}'.format(goal_x), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, 'Goal_y: {:.2f}'.format(goal_y), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, 'Pred dist: {:.2f}'.format(predicted_distance), (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)

        cv2.putText(frame, 'Goal # {}{}{}'.format(goal_num_text,
                                                  op,
                                                  total_goals),
                                                  (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, 'Success rate: {}/{} {:.2f}%'.format(self.successful_goal_completions,
                                                                total_goals_passed,
                                                                self.completion_rate * 100), 
                                                                (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, 'Diff in odom dist: {:.2f}'.format(np.linalg.norm(self.prev_pose - self.cur_pose)), (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, 'Model: {}'.format(self.local_model_path.split('/')[-1]), (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, 'Drive Mode: {}'.format(self.mode), (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        
        cv2.putText(frame, '# of Interventions: {}'.format(self.num_interventions), (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
    

    
    def joy_callback(self, joy_msg):
        # check the button press if in automatic/manual mode
        # if manual mode button press        
        if joy_msg.buttons[4] == 1.0 or joy_msg.buttons[5] == 1.0:
            # if the mode previously was automatic, then increment the # of interventions by 1
            # set the mode to manual
            if self.mode == "automatic":
                self.num_interventions += 1 
                self.mode = "manual"
        
        # else automatic
        else:
            self.mode = "automatic"        

        # toggle goal images from joystick left/right buttons for previous/next goal img respectively
        # toggle to previous goal img
        if joy_msg.buttons[11] == 1.0:
            if self.goal_img_index > 0:
                    self.goal_img_index -= 1
            self.prev_pose = self.cur_pose
            time.sleep(0.5)

        # toggle to next goal img
        elif joy_msg.buttons[12] == 1.0:
            if self.goal_img_index < self.sampled_img_batch.shape[0]-1:
                    self.goal_img_index += 1
            self.prev_pose = self.cur_pose
            time.sleep(0.5)

        # exit the node if pressed triangle button
        elif joy_msg.buttons[3] == 1.0:
            self.final_goal_passed = True
            self.video.release()
            time.sleep(1)            
            print(f"User enabled shutdown")                
            rospy.signal_shutdown("User enabled shutdown")


    def odom_callback(self, odom_msg):
        self.cur_pose = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])

        if self.prev_pose is None:
            self.prev_pose = self.cur_pose
    
    def image_callback(self, img_msg):

        self.goal_num = self.goal_img_index + 1

        self.goal_img = self.sampled_img_batch[self.goal_img_index]
        self.goal_img_resized = center_crop_and_resize(self.goal_img, self.img_size)
        self.goal_img_preprocessed = prepare_image(self.goal_img_resized)

        img = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        self.deque_images.append(img)

        if len(self.deque_images) < self.buffer_length:
            driving_command = Twist()
            driving_command.linear.x = 0
            driving_command.angular.z = 0
            self.driving_command_publisher.publish(driving_command)            
            return
        
        self.obs_img = batch_obs_plus_context(buffer_length = self.buffer_length,
                                            waypoint_spacing = self.waypoint_spacing,
                                            deque_images = self.deque_images, 
                                            fps = self.fps, 
                                            crop_size = self.img_size
                                        )
        

        forward_model_in = {'obs_img': self.obs_img,
                            'goal_img': self.goal_img_preprocessed}
        forward_model_out = self.local_session.run(None, forward_model_in)
        
        predicted_distance = np.squeeze(forward_model_out[0])
        predicted_actions = np.squeeze(forward_model_out[1])

        if self.local_model_type == "vae":
            waypoint_samples = np.squeeze(forward_model_out[3])
        
        waypoints_in_optical_frame = np.zeros((self.waypoint_length, 3))
        for i in range(self.waypoint_length):
            wp_in_cam_frame = PointStamped()
            wp_in_cam_frame.header.stamp = rospy.Time.now().to_sec
            wp_in_cam_frame.point.x = predicted_actions[i][0]
            wp_in_cam_frame.point.y = predicted_actions[i][1]         
            wp_in_cam_frame.point.z = -0.5         

            goal_in_optical_frame = tf2_geometry_msgs.do_transform_point(wp_in_cam_frame, self.tf_from_cam_frame_to_optical_frame)
            
            waypoints_in_optical_frame[i][0] = goal_in_optical_frame.point.x
            waypoints_in_optical_frame[i][1] = goal_in_optical_frame.point.y
            waypoints_in_optical_frame[i][2] = goal_in_optical_frame.point.z            
        
        
        goal_x, goal_y = predicted_actions[-1][:2]
        self.goal_in_left_camera_frame.header.stamp = rospy.Time.now().to_sec
        self.goal_in_left_camera_frame.point.x = goal_x
        self.goal_in_left_camera_frame.point.y = goal_y

        self.goal_in_base_link = tf2_geometry_msgs.do_transform_point(self.goal_in_left_camera_frame, self.tf_from_cam_frame_to_base_link)

        goal_x = self.goal_in_base_link.point.x
        goal_y = self.goal_in_base_link.point.y

        dist = np.sqrt((goal_x**2) + (goal_y**2))
        theta = np.arctan2(goal_y, goal_x)
        
        dist_from_prev_pose = np.linalg.norm(self.prev_pose - self.cur_pose)
        
        v = dist
        w = theta

        v = np.clip(v, 0, 0.2)
        w = np.clip(w, -0.3, 0.3)
        
        self.driving_command.linear.x = v
        self.driving_command.angular.z = w

        self.driving_command_publisher.publish(self.driving_command)
        
        
        rectified_img = rectify_image(self.camera_model, img)
        draw_waypoint_image(rectified_img, self.goal_img_resized)

        visualize_waypoints(rectified_img, 
                            predicted_actions, 
                            predicted_actions.shape[1], 
                            self.tf_from_cam_frame_to_optical_frame, 
                            self.camera_model, 
                            color=RED, # RED 
                            radius=5)

        visualization_init_time = time.time()

        # if its the last goal image
        if self.goal_img_index == self.sampled_img_batch.shape[0] - 1:
            
            # final goal conditioned
            if predicted_distance <= self.goal_conditioning_threshold: # off-policy
                self.successful_goal_completions += 1
                self.final_goal_passed =True
            
            # if the final goal has not been conditioned
            else:
                # if the robot exceeds the odom bounds
                if dist_from_prev_pose >= self.goals_spacing + self.odometry_tolerance: 
                    self.final_goal_passed = True


        # if it is not the last goal image
        elif self.goal_img_index < self.sampled_img_batch.shape[0] - 1:
                
            # confirming goal conditioning
            if predicted_distance <= self.goal_conditioning_threshold:
                self.prev_pose = self.cur_pose
                self.successful_goal_completions += 1
                self.goal_img_index += 1
            
            # if goal not conditioned
            else:
                # if the robot exceeds the odometry bounds from previous goal or starting position
                if dist_from_prev_pose >= self.goals_spacing + self.odometry_tolerance:
                    self.prev_pose = self.cur_pose
                    self.goal_img_index += 1


        self.draw_info_overlay(frame=rectified_img,
                                predicted_distance=predicted_distance,
                                goal_x=goal_x,
                                goal_y=goal_y,
                                v=v,
                                w=w)

        draw_top_down_overlay(rectified_img, predicted_actions[:, 0:2])
        show_next_frame(rectified_img)
        
        if self.record_video:
            self.video.write(rectified_img)

        if self.final_goal_passed == True:
            if self.record_video:
                for _ in range(50):
                    self.video.write(rectified_img)
                self.video.release()
            rospy.signal_shutdown("Final goal conditioned or user enabled shutdown")

        key = cv2.waitKey(1)

        if key == ord('n'):
            if self.goal_img_index < self.sampled_img_batch.shape[0]-1:
                self.goal_img_index += 1
        elif key == ord('p'):
            if self.goal_img_index > 0:
                self.goal_img_index -= 1
        elif key == 27: # 'ESC press'
            if self.record_video:
                for _ in range(50):
                    self.video.write(rectified_img)
                self.video.release()
            cv2.destroyAllWindows()
            rospy.signal_shutdown("User pressed ESC")


if __name__ == "__main__":
    rospy.init_node("Odom_Based_Goal_Image_Update_Inference_Visualization", log_level=rospy.INFO)
    node = Visualizer()
    rospy.spin()

    