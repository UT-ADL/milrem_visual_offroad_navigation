import math

import cv2


def draw_direction(position, north_heading, map_img, length=10, thickness=1, color=(255, 0, 0)):
    draw_direction_alpha(position, north_heading-90, map_img, length, thickness, color)


def draw_direction_alpha(position, alpha, map_img, length=10, thickness=1, color=(255, 0, 0)):
    alpha_rad = math.radians(alpha)
    x_delta = length * math.cos(alpha_rad)
    y_delta = length * math.sin(alpha_rad)
    next_position = (position[0] + int(x_delta), position[1] + int(y_delta))
    cv2.line(map_img, position, next_position, color, thickness)


def calculate_angle(A, B):
    radian = math.atan2(B[1] - A[1], B[0] - A[0])
    degree = math.degrees(radian)
    return degree


def calculate_north_heading(A, B):
    return calculate_angle(A, B) + 90


def calculate_distance(A, B):
    return math.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)