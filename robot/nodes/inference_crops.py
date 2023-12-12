import os
import math
import yaml

from collections import deque

import cv2
import numpy as np

import onnxruntime as ort

import rospy
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel

import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion

from sensor_msgs.msg import CompressedImage, NavSatFix, CameraInfo, Image
from geometry_msgs.msg import Twist, PointStamped, Vector3Stamped, QuaternionStamped
from robot.msg import ImageArray

from global_planner.data.mapping import MapReader
from global_planner.models.global_planner_onnx import GlobalPlannerOnnx
from global_planner.viz.global_planner_viz import GlobalPlannerViz
from global_planner.data.util import normalize_image

from helpers.timer import Timer

from utils.viz_util import draw_rectangle, show_next_frame, rectify_image, visualize_waypoints, draw_trapezoids, draw_waypoint_image, draw_waypoint_images
from utils.preprocessing_images import center_crop_and_resize, prepare_image
from utils.batching import batch_obs_plus_context

# Color constants, in RGB order
BLUE = (0, 0, 255)
RED  = (255, 0, 0)
GREEN = (0, 128, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 102, 0)
CYAN = (0, 255, 255)
FONT_SIZE = 0.3

TRAPEZOIDS = [
    [[0, 0], [399, 30], [399, 329], [0, 359]],
    [[60, 30], [459, 60], [459, 299], [60, 329]],
    [[120, 60], [519, 60], [519, 299], [120, 299]],
    [[180, 60], [579, 30], [579, 329], [180, 299]],
    [[240, 30], [639, 0], [639, 359], [240, 329]]
]

# Trajectory colors, in BGR order
TRAJ_COLORS = (
    (255, 0, 0),    # BLUE
    (0, 255, 0),    # GREEN
    (0, 0, 255),    # RED
    (0, 0, 0),      # BLACK
    (255, 0, 255),  # MAGENTA
    (255, 255, 255) # WHITE
)


class Visualizer:

    def __init__(self):

        # Fetch parameters
        self.config_dir_path = rospy.get_param('~config_dir_path')
        self.local_model_type = rospy.get_param('~local_model_type')
        self.local_model_path = rospy.get_param('~local_model_path')
        self.global_model_path = rospy.get_param('~global_model_path')
        self.map_path = rospy.get_param('~map_path')
        self.map_name = rospy.get_param('~map_name')
        self.fps = int(rospy.get_param('~fps'))
        self.orientation_fix = rospy.get_param('~orientation_fix')
        self.goal_conditioning_threshold = rospy.get_param('~goal_conditioning_threshold')
        self.base_link_frame = rospy.get_param('~base_link_frame', 'base_link_frame')
        self.left_camera_frame = rospy.get_param('~left_camera_frame', 'zed_left_camera_frame')
        self.left_camera_optical_frame = rospy.get_param('~left_camera_optical_frame', 'zed_left_camera_optical_frame')
        self.record_video = rospy.get_param('~record_video')
        self.timer_frequency = rospy.get_param('~timer_frequency')

        rospy.loginfo(f"fps: {self.fps, type(self.fps)}")

        # Initialize tf2 listener and CV bridge
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()

        # Load local planner configration
        local_planner_config_file = os.path.join(self.config_dir_path, str(self.local_model_type) + '.yaml')
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
        
        # Load global planner configration
        global_planner_config_file = os.path.join(self.config_dir_path, 'distance_segment.yaml')
        with open(global_planner_config_file, "r") as f:
            self.global_planner_config = yaml.safe_load(f)
        rospy.loginfo(f"Loaded global planner config: {global_planner_config_file}")

        # Load global planner map
        map_type = self.global_planner_config["map_type"]
        map_file_path = os.path.join(self.map_path, f"{self.map_name}_{map_type}.tif")
        self.map_reader = MapReader(map_file_path, self.global_planner_config["map_size"])
        rospy.loginfo(f"Loaded global planner map: {map_file_path}")

        # Load global planner model
        self.global_planner = GlobalPlannerOnnx(self.map_reader, self.global_planner_config, self.global_model_path)
        rospy.loginfo(f"Loaded global planner model: {self.global_model_path}")

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

        # Prepare perspective transform matrices for each crop
        self.target_points = np.float32([[0, 0], [self.img_size[0]-1, 0], [self.img_size[0]-1, self.img_size[1]-1], [0, self.img_size[1]-1]])
        self.matrices = []
        for i in range(len(TRAPEZOIDS)):
            matrix = cv2.getPerspectiveTransform(np.float32(TRAPEZOIDS[i]), self.target_points)
            self.matrices.append(matrix)

        # Initialize video writer
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # self.video = cv2.VideoWriter(self.record_video, fourcc, self.fps, (640, 360))
            self.video = cv2.VideoWriter(self.record_video, fourcc, self.timer_frequency, (640, 360))
            rospy.loginfo(f"Recording video to: {self.record_video}")

        # Initialize internal variables
        self.obs_img = None
        self.goal_img = None        

        self.goal_img_resized = center_crop_and_resize(np.zeros((360, 640, 3), dtype=np.uint8), self.img_size)
        self.goal_img_preprocessed = prepare_image(self.goal_img_resized)

        self.goal_gps = None
        self.current_gps = None
        self.current_heading = None

        num_crops = len(TRAPEZOIDS)
        self.distance_predictions = np.zeros(num_crops + 1)
        self.waypoint_predictions = np.zeros((num_crops + 1, self.waypoint_length , 2))
        self.show_crops = False

        # Fetch transformers
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
        
        rospy.Timer(rospy.Duration(1.0/self.timer_frequency), self.timer_callback)

        rospy.Subscriber('/goal_image',
                        Image,
                        self.goal_image_callback)

        rospy.Subscriber('/goal_gps',
                        NavSatFix,
                        self.goal_gps_callback)

        rospy.Subscriber('/filter/positionlla',
                        Vector3Stamped,
                        self.current_gps_callback)

        rospy.Subscriber('/filter/quaternion',
                        QuaternionStamped,
                        self.current_heading_callback)

        rospy.Subscriber('/zed/zed_node/left_raw/image_raw_color/compressed',
                        CompressedImage,
                        self.image_callback,
                        queue_size=1,
                        buff_size=2**24)
    
    def draw_info_overlay(self, frame, predicted_distance, goal_x, goal_y, v, w):
        
        draw_rectangle(frame, 5, 100, 200, 200)

        cv2.putText(frame, 'Driving Commands:', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, '  Linear_X: {:.2f}'.format(v), (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, '  Angular_Z: {:.2f}'.format(w), (15, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Goal_x: {:.2f}'.format(goal_x), (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Goal_y: {:.2f}'.format(goal_y), (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, 'Pred dist: {:.2f}'.format(predicted_distance), (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, GREEN, 1, cv2.LINE_AA)
    

    def draw_global_map(self, frame, relative_waypoints):
        trajectories = relative_waypoints[:, np.newaxis, :]
        global_planner_viz = GlobalPlannerViz(self.global_planner)
        cropped_map = global_planner_viz.plot_trajectories_map(self.current_gps, self.goal_gps, self.current_heading, trajectories, TRAJ_COLORS)
        probability_map_img = global_planner_viz.plot_probability_map(self.current_gps, self.goal_gps)

        # Crop images
        crop_half = (50, 50)
        map_center = (cropped_map.shape[0] // 2, cropped_map.shape[1] // 2)
        cropped_map = cropped_map[map_center[0] - crop_half[0]:map_center[0] + crop_half[0], map_center[1] - crop_half[1]:map_center[1] + crop_half[1]]
        cropped_map = cv2.cvtColor(cropped_map, cv2.COLOR_RGB2BGR)
        probability_map_img = probability_map_img[map_center[0] - crop_half[0]:map_center[0] + crop_half[0], map_center[1] - crop_half[1]:map_center[1] + crop_half[1]]
        probability_map_img = cv2.cvtColor(probability_map_img, cv2.COLOR_RGB2BGR)

        # Put both maps side-by-side to the bottom right of the frame
        frame[frame.shape[0]-cropped_map.shape[0]:, frame.shape[1]-cropped_map.shape[1]:] = cropped_map
        frame[frame.shape[0]-probability_map_img.shape[0]:, frame.shape[1]-probability_map_img.shape[1]-cropped_map.shape[1]:frame.shape[1]-cropped_map.shape[1]] = probability_map_img


    def goal_image_callback(self, goal_img_msg):
        self.goal_img = self.bridge.imgmsg_to_cv2(goal_img_msg)
        rospy.loginfo(f"Goal image received: {self.goal_img.shape}")

        
        # Resize image for display, use the center crop
        self.goal_img_resized = center_crop_and_resize(self.goal_img, self.img_size)
        rospy.loginfo(f"Goal image resized : {self.goal_img_resized.shape}")
        # Preprocess goal image                
        self.goal_img_preprocessed = prepare_image(self.goal_img_resized)
        

    def goal_gps_callback(self, gps_msg):
        self.goal_gps = (gps_msg.latitude, gps_msg.longitude)

    def current_gps_callback(self, gps_msg):
        # Record current GPS position
        self.current_gps = (gps_msg.vector.x, gps_msg.vector.y)
        #print("current_gps:", self.current_gps)

    def current_heading_callback(self, quaternion_msg):
        quaternion = quaternion_msg.quaternion
        roll, pitch, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.current_heading = (-math.degrees(yaw) + self.orientation_fix) % 360
        #print("current_heading:", self.current_heading)

    def image_callback(self, img_msg): 
        
        # Prepare observation image
        img = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        self.deque_images.append(img)
        
    
    def timer_callback(self, event=None):

        # Stop if buffer is not full yet or no goal image
        if len(self.deque_images) < self.buffer_length or self.goal_img is None:
            driving_command = Twist()
            driving_command.linear.x = 0
            driving_command.angular.z = 0
            self.driving_command_publisher.publish(driving_command)
            rospy.loginfo_throttle(5, "No goal image - stopping the robot")
            return

        else:            
            img = self.deque_images[-1]

            # Prepare goal images
            goal_images_preprocessed = []
            goal_images_resized = []
            for i in range(len(TRAPEZOIDS)):
                # Crop goal from observation image
                crop = cv2.warpPerspective(img, self.matrices[i], tuple(self.img_size))            
                crop_resized = center_crop_and_resize(crop, self.img_size)                
                crop_preprocessed = prepare_image(crop_resized)            
                
                goal_images_resized.append(crop_resized)
                goal_images_preprocessed.append(crop_preprocessed)
                
            # Add final goal imagebatch_obs_plus_context
            goal_images_resized.append(self.goal_img_resized)
            goal_images_preprocessed.append(self.goal_img_preprocessed)
            goal_images_preprocessed = np.array(goal_images_preprocessed)
            
            
            obs_img = batch_obs_plus_context(buffer_length = self.buffer_length,
                                             waypoint_spacing = self.waypoint_spacing,
                                             deque_images = self.deque_images, 
                                             fps = self.fps, 
                                             crop_size = self.img_size
                                            )
            

            # Run inference for all goal images
            # TODO: Should be done in a batch
            for i in range(len(goal_images_preprocessed)):

                local_model_in = {'obs_img': obs_img,
                                'goal_img': goal_images_preprocessed[i]
                                }
                local_model_out = self.local_session.run(None, local_model_in)

                self.distance_predictions[i] = local_model_out[0]
                self.waypoint_predictions[i] = local_model_out[1][0, :, 0:2]

            # Check if final goal is reached
            if self.distance_predictions[-1] <= self.goal_conditioning_threshold:
                # This should stop the robot in the next iteration
                self.goal_img = None
                self.goal_gps = None
                rospy.loginfo("Final goal reached")
                return

            # Take the last point from all trajectories
            candidate_waypoints = self.waypoint_predictions[:, -1, :]

            # Choose the waypoint to follow
            if self.current_gps is not None and self.current_heading is not None and self.goal_gps is not None:
                # Create probability map
                self.global_planner.predict_probabilities(self.current_gps, self.goal_gps)                

                # Calculate probabilities for each waypoint
                probs = self.global_planner.calculate_probs(candidate_waypoints, self.current_gps, self.current_heading)                

                # Pick the waypoint with the highest probability
                best_waypoint = np.argmax(probs)
            else:
                best_waypoint = -1 # the final goal trajectory
            

            # Drive towards the last waypoint of the best trajectory
            goal_x, goal_y = self.waypoint_predictions[best_waypoint, -1]

            # Convert the waypoint from camera frame to base_link
            goal_in_left_camera_frame = PointStamped()
            goal_in_left_camera_frame.header.stamp = rospy.Time.now().to_sec
            goal_in_left_camera_frame.point.x = goal_x
            goal_in_left_camera_frame.point.y = goal_y

            goal_in_base_link = tf2_geometry_msgs.do_transform_point(goal_in_left_camera_frame, self.tf_from_cam_frame_to_base_link)

            # Calculate speed and angular velocity
            goal_x , goal_y = goal_in_base_link.point.x, goal_in_base_link.point.y 
            dist = np.sqrt((goal_x**2) + (goal_y**2))            
            theta = np.arctan2(goal_y, goal_x)

            v = dist
            w = theta

            v = np.clip(v, 0, 0.2)
            w = np.clip(w, -0.4, 0.4)

            # Publish driving command
            driving_command = Twist()
            driving_command.linear.x = v
            driving_command.angular.z = w

            self.driving_command_publisher.publish(driving_command)
            

            # Rectify the camera image to be able to project waypoints onto it
            rectified_img = rectify_image(self.camera_model, img)
            
            # Draw the trapezoids used for goal crops
            if self.show_crops:
                draw_trapezoids(rectified_img, TRAPEZOIDS, TRAJ_COLORS)
                # Draw all the goal images
                draw_waypoint_images(rectified_img, goal_images_resized, TRAJ_COLORS)
            else:
                # Draw the final goal image
                draw_waypoint_image(rectified_img, self.goal_img_resized)
            
            # Draw the waypoints
            for i in range(len(self.waypoint_predictions)):
                visualize_waypoints(rectified_img, self.waypoint_predictions[i], self.waypoint_length, self.tf_from_cam_frame_to_optical_frame, self.camera_model, TRAJ_COLORS[i], radius=5)

            # Visualize global map
            if self.current_gps is not None and self.current_heading is not None and self.goal_gps is not None:
                self.draw_global_map(rectified_img, candidate_waypoints)

            self.draw_info_overlay(frame=rectified_img,
                                    predicted_distance=self.distance_predictions[-1],
                                    goal_x=goal_x,
                                    goal_y=goal_y,
                                    v=v,
                                    w=w)

            show_next_frame(rectified_img)
            if self.record_video:
                self.video.write(rectified_img)

            key = cv2.waitKey(1)
            if key == 27:
                if self.record_video:
                    self.video.release()
                cv2.destroyAllWindows()
                rospy.signal_shutdown("User pressed ESC")
            elif key == ord('c'):
                self.show_crops = not self.show_crops

            


if __name__ == "__main__":
    rospy.init_node("Current_Image_Crop_Goal_Update_Inference_Visualization", log_level=rospy.INFO)
    node = Visualizer()    
    rospy.spin()
