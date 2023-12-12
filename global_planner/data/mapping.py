import math

import numpy as np
import rasterio
from pyproj import CRS, Proj, Transformer
from rasterio.enums import Resampling


class MapReader:

    def __init__(self, map_path, crop_size) -> None:
        with rasterio.open(map_path) as dataset:
            self.map_resolution = np.array(dataset.res)
            # resample data to target shape
            self.map_data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * dataset.res[0]),
                    int(dataset.width * dataset.res[1])
                ),
                resampling=Resampling.bilinear
            )

            # scale image transform
            self.map_transform = dataset.transform * dataset.transform.scale(
                (dataset.width / self.map_data.shape[-1]),
                (dataset.height / self.map_data.shape[-2])
            )
        self.map_img = np.transpose(self.map_data, axes=[1, 2, 0]).astype(np.uint8).copy()
        self.map_size = crop_size

        self.transformer = Transformer.from_crs("epsg:4326", "epsg:25835")
        self.utm_crs = Proj(CRS.from_epsg(25835))  # Europe zone 35N that includes Estonia

    def to_px(self, position):
        return self.lat_lon_to_pixel([position[0]], [position[1]])[0]

    def lat_lon_to_pixel(self, lat, lon):
        x, y = self.transformer.transform(lat, lon)
        positions_in_pixels = [~self.map_transform * (x, y) for x, y in zip(x, y)]
        return np.array(positions_in_pixels).astype(int)

    def adjust_heading(self, lat, lon, heading_degrees):
        """Adjusts heading from magnetic north to geographic north."""
        heading_offset = self.utm_crs.get_factors(lon, lat).meridian_convergence
        heading_degrees += heading_offset
        return heading_degrees

    def convert_waypoints_to_pixel_coordinates(self, current_position, heading_degrees, relative_waypoints):
        # From meters to pixels
        relative_waypoints_pix = relative_waypoints / self.map_resolution
        waypoints_in_pixels = convert_trajectory(current_position, heading_degrees, relative_waypoints_pix)
        return waypoints_in_pixels

    def map_window(self, center_x, center_y, map_size=200):
        center_x_px, center_y_px = ~self.map_transform * (center_x, center_y)

        x_start = int(center_x_px - map_size)
        x_end = int(center_x_px + map_size)

        y_start = int(center_y_px - map_size)
        y_end = int(center_y_px + map_size)

        map_window = self.map_img[y_start:y_end, x_start:x_end]
        return map_window

    def crop_map_by_position(self, position):
        center_x, center_y = int(position[0]), int(position[1])
        x_start = max(center_x - self.map_size, 0)
        x_end = center_x + self.map_size
        y_start = max(center_y - self.map_size, 0)
        y_end = center_y + self.map_size
        map_window = self.map_img[y_start:y_end, x_start:x_end]
        map_window = map_window[:, :, :3]

        x_start_position = 0
        if center_x - self.map_size < 0:
            x_start_position = abs(center_x - self.map_size)

        y_start_position = 0
        if center_y - self.map_size < 0:
            y_start_position = abs(center_y - self.map_size)

        x_end_position = x_start_position + map_window.shape[1]
        y_end_position = y_start_position + map_window.shape[0]

        padded_map = np.zeros((2*self.map_size, 2*self.map_size, 3), dtype=np.uint8)
        padded_map[y_start_position:y_end_position, x_start_position:x_end_position, :] = map_window

        return padded_map

    def to_crop_coordinates(self, current_position, position):
        local_position = position - current_position + self.map_size
        return int(local_position[0]), int(local_position[1])


def rotate_point(x, y, angle):
    """Rotates a point counterclockwise by a given angle around the origin."""
    radians = math.radians(angle)
    x_new = x * math.cos(radians) + y * math.sin(radians)
    y_new = x * math.sin(radians) - y * math.cos(radians)

    return x_new, y_new


def convert_trajectory(current_position, heading, trajectory):
    """Converts a relative trajectory to absolute coordinates."""
    absolute_trajectory = []
    for rel_x, rel_y in trajectory:
        # Rotate the relative displacement by the heading angle
        # The heading is measured from the north, so we need to subtract it from 90 degrees
        # y is negated as weirdly local planner encodes positive y to the left and negative y to the right.
        rotated_x, rotated_y = rotate_point(rel_x, -rel_y, 90 - heading)
        # Translate the point by the current position
        abs_x = current_position[0] + rotated_x
        abs_y = current_position[1] - rotated_y
        absolute_trajectory.append((abs_x, abs_y))
    return np.array(absolute_trajectory).astype(int)