import numpy as np

EPSILON = 1e-10  # to avoid divide by zero error


class Mask:

    def __init__(self, distance_threshold=3, mask_type='simple', kernel_size=33, sigma=3.1):
        self.distance_threshold = distance_threshold
        self.mask_type = mask_type
        self.kernel_size = kernel_size
        self.sigma = sigma

    def create_masked_map(self, map_tensor, trajectory):
        mask_shape = map_tensor.shape[1:]
        start_mask = np.expand_dims(self.create_mask(mask_shape, trajectory[[0]]), axis=0)
        end_mask = np.expand_dims(self.create_mask(mask_shape, trajectory[[-1]]), axis=0)
        masked_map = np.concatenate([map_tensor, start_mask, end_mask], axis=0)
        return masked_map

    def create_mask(self, mask_shape, trajectory):
        x, y = np.meshgrid(np.arange(mask_shape[0]), np.arange(mask_shape[1]), indexing='xy')
        distances = np.sqrt((x - np.array(trajectory)[:, 0][:, np.newaxis, np.newaxis]) ** 2 +
                            (y - np.array(trajectory)[:, 1][:, np.newaxis, np.newaxis]) ** 2)

        mask = (distances <= self.distance_threshold).any(axis=0).astype(np.float32)

        if self.mask_type == 'gaussian':
            mask = create_gaussian_mask(mask, self.kernel_size, self.sigma)
        elif self.mask_type == 'distance':
            mask = create_distance_mask(mask, trajectory)
        elif self.mask_type == "gaussian-distance":
            mask = create_gaussian_mask(mask, self.kernel_size, self.sigma)
            mask = create_distance_mask(mask, trajectory)

        return mask.astype(np.float32)


def create_distance_tensor(goal_tensor):
    # Finding all positions with the value 1
    positions = np.column_stack(np.where(goal_tensor == 1))

    # Create coordinate grids
    xx, yy = np.meshgrid(np.arange(goal_tensor.shape[0]), np.arange(goal_tensor.shape[1]), indexing='xy')

    # Reshape the grid coordinates for broadcasting
    xx_r = xx[:, :, np.newaxis]
    yy_r = yy[:, :, np.newaxis]

    # Compute squared distances using broadcasting
    squared_differences = (xx_r - positions[:, 0]) ** 2 + (yy_r - positions[:, 1]) ** 2

    # Take the square root and then find the minimum distance along the last dimension
    distance_tensor = np.sqrt(np.min(squared_differences, axis=2)).astype(np.float32)

    return distance_tensor


def limit_trajectory(trajectory, shape):
    """Shortens trajectories if they are outside of the map area."""
    trajectory_end = len(trajectory)
    for i in range(len(trajectory)):
        goal_x, goal_y = trajectory[i]
        if goal_x < 0 or goal_x >= shape[0] or goal_y < 0 or goal_y >= shape[1]:
            # substract one so the end of the trajectory would not be exactly on the edge of the map
            trajectory_end = i - 1
            # trajectory must have atleast current and goal positions
            trajectory_end = max(trajectory_end, 2)
            break
    result = trajectory[:trajectory_end, :]
    return result


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def create_distance_mask(trajectory_mask, trajectory):
    goal_tensor = np.zeros(trajectory_mask.shape)
    goal = trajectory[-1]
    goal_x = clamp(int(goal[0]), 0, trajectory_mask.shape[0]-1)
    goal_y = clamp(int(goal[1]), 0, trajectory_mask.shape[1]-1)

    goal_tensor[goal_x][goal_y] = 1
    distance_tensor = create_distance_tensor(goal_tensor)

    max_distance = distance_tensor.max()
    distance_mask = 1 - (trajectory_mask * (distance_tensor - max_distance) / max_distance)
    max = distance_mask.max()
    min = distance_mask.min()
    distance_mask = (distance_mask - min) / (max - min + EPSILON)
    return distance_mask


def gaussian_2d_kernel(kernel_size, sigma):
    """Generates a 2D Gaussian kernel."""
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(x, x, indexing='xy')
    kernel = np.exp(- (xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def create_gaussian_mask(map_tensor, kernel_size, sigma):
    """Applies a 2D Gaussian distribution around each trajectory point."""

    # Generate the Gaussian kernel
    kernel = gaussian_2d_kernel(kernel_size, sigma)

    # Prepare a tensor to store the result
    result = np.zeros_like(map_tensor, dtype=float)

    # Get the coordinates of all trajectory points
    y, x = np.where(map_tensor == 1)

    half_k = kernel_size // 2

    # Apply the Gaussian kernel around each trajectory point
    for xi, yi in zip(x, y):
        x_min, x_max = max(0, xi - half_k), min(map_tensor.shape[1], xi + half_k + 1)
        y_min, y_max = max(0, yi - half_k), min(map_tensor.shape[0], yi + half_k + 1)

        k_x_min, k_x_max = half_k - (xi - x_min), half_k + (x_max - xi)
        k_y_min, k_y_max = half_k - (yi - y_min), half_k + (y_max - yi)

        result[y_min:y_max, x_min:x_max] += kernel[k_y_min:k_y_max, k_x_min:k_x_max]

    # Normalize to [0, 1]

    result = (result - result.min()) / (result.max() - result.min() + EPSILON)

    return result