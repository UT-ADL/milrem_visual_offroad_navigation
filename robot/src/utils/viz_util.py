import cv2
import numpy as np

import rospy
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs


# top-down overlay + top-down trajectory
# visualize waypoints
# draw waypoint image
# draw rectangle
# show next frame
# rectify image

RED = (0,0, 255)

def draw_rectangle(frame, x, y, w, h):
    sub_img = frame[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, (0, 0, 0), np.uint8)
    alpha = 0.4
    res = cv2.addWeighted(sub_img, alpha, rect, 1-alpha, 0)
    frame[y:y+h, x:x+w] = res


def show_next_frame(img):
    img_resized = cv2.resize(img, (2*img.shape[1], 2*img.shape[0]), interpolation=(cv2.INTER_AREA))
    cv2.imshow('Live Camera Frame', img_resized)


def rectify_image(camera_model, img):
    rectified_image = np.empty_like(img)
    camera_model.rectifyImage(raw=img, rectified=rectified_image)
    return rectified_image


def draw_top_down_overlay(frame, predicted_wps):
    TOP_DOWN_VIEW_SCALE = 15
    draw_rectangle(frame, 5, 245, 160, 160)    
    current_position = (75, 350)

    predicted_wps = predicted_wps.reshape(-1, 2)
    draw_top_down_trajectory(frame, predicted_wps, current_position, TOP_DOWN_VIEW_SCALE, RED)

    cv2.circle(frame, current_position, 2, (0, 255, 255), 2)

def draw_top_down_trajectory(frame, predicted_wps, current_position, TOP_DOWN_VIEW_SCALE, color):

    for i in range(predicted_wps.shape[0]):
        pred_wp = predicted_wps[i]
        
        scaled_current_wp = (current_position[0] - int(TOP_DOWN_VIEW_SCALE * pred_wp[1]),
                                current_position[1] - int(TOP_DOWN_VIEW_SCALE * pred_wp[0]))
        cv2.circle(frame, scaled_current_wp, 2, color, 2)

        if i > 0:
            pred_prev_wp = predicted_wps[i - 1]
            scaled_prev_wp = (current_position[0] - int(TOP_DOWN_VIEW_SCALE * pred_prev_wp[1]),
                                current_position[1] - int(TOP_DOWN_VIEW_SCALE * pred_prev_wp[0]))
            cv2.line(frame, scaled_prev_wp, scaled_current_wp, color, 1)
    
    scaled_first_wp = (current_position[0] - int(TOP_DOWN_VIEW_SCALE * predicted_wps[0][1]),
                       current_position[1] - int(TOP_DOWN_VIEW_SCALE * predicted_wps[0][0]))
    cv2.line(frame, current_position, scaled_first_wp, color, 1)



def draw_waypoint_image(frame, goal_img):
    spacing = 10
    frame[spacing:goal_img.shape[0]+spacing, frame.shape[1]-goal_img.shape[1]-spacing:frame.shape[1]-spacing] = goal_img


def draw_waypoint_images(frame, goal_imgs, trajectory_colors):
    spacingx = 20
    spacingy = 10
    for i, goal_img in enumerate(goal_imgs):        
        start = spacingx + i * (goal_img.shape[1] + spacingx)
        frame[spacingy:goal_img.shape[0] + spacingy, start:start + goal_img.shape[1]] = goal_img
        cv2.rectangle(frame, (start - 1, spacingy - 1), (start + goal_img.shape[1] + 1, goal_img.shape[0] + spacingy + 1), trajectory_colors[i], 2)


def draw_trapezoids(frame, trapezoids, trajectory_colors):
        for i in range(len(trapezoids)):
            cv2.polylines(frame, [np.array(trapezoids[i])], True, trajectory_colors[i], 2)



def visualize_waypoints(img, predicted_actions, waypoint_length, transform, camera_model, color, radius=5):
    trajectory_in_optical_frame = np.zeros((waypoint_length, 3))
    # color = (color[2], color[1], color[0])
    for i in range(waypoint_length):
        wp_in_cam_frame = PointStamped()
        wp_in_cam_frame.header.stamp = rospy.Time.now()
        wp_in_cam_frame.point.x = predicted_actions[i][0]
        wp_in_cam_frame.point.y = predicted_actions[i][1]
        wp_in_cam_frame.point.z = -0.5

        goal_in_optical_frame = tf2_geometry_msgs.do_transform_point(wp_in_cam_frame,
                                                                     transform)
        
        trajectory_in_optical_frame[i][0] = goal_in_optical_frame.point.x
        trajectory_in_optical_frame[i][1] = goal_in_optical_frame.point.y
        trajectory_in_optical_frame[i][2] = goal_in_optical_frame.point.z

    camera_pixel_coords = np.zeros(shape=(trajectory_in_optical_frame.shape[0], 2))
    z = np.zeros(trajectory_in_optical_frame.shape[0])

    for id, point in enumerate(trajectory_in_optical_frame):
        point3d = point[:3]
        z[id] = point3d[-1]
        camera_pixel_coords[id] = camera_model.project3dToPixel(point3d)

    for i, pixel_coord in enumerate(camera_pixel_coords):
        # if i > 0:
        scaled_radius = int(np.abs((1 / (z[i]+0.001) * radius)))
        current_wp = (int(pixel_coord[0]), int(pixel_coord[1]))
        cv2.circle(img, current_wp, scaled_radius, color, 2)
        
        if i > 0:
            prev_wp = (int(camera_pixel_coords[i-1][0]), int(camera_pixel_coords[i-1][1]))
            cv2.line(img, prev_wp, current_wp, color, 2)