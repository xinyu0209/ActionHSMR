from lib.kits.basic import *

# Copied from: https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/datasets/utils.py#L663-L944

def crop_to_hips(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray) -> Tuple:
    """
    Extreme cropping: Crop the box up to the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24, 25+0, 25+1, 25+4, 25+5]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_to_shoulders(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box up to the shoulder locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    center, scale = get_bbox(keypoints_2d)
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.2 * scale[0]
        height = 1.2 * scale[1]
    return center_x, center_y, width, height


def crop_to_head(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the head.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.3 * scale[0]
        height = 1.3 * scale[1]
    return center_x, center_y, width, height


def crop_torso_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the torso.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nontorso_body_keypoints = [0, 3, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 4, 5, 6, 7, 10, 11, 13, 17, 18]]
    keypoints_2d[nontorso_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_rightarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the right arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightarm_body_keypoints = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_leftarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the left arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftarm_body_keypoints = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_legs_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the legs.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonlegs_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18] + [25 + i for i in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]]
    keypoints_2d[nonlegs_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_rightleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the right leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightleg_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] + [25 + i for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.ndarray):
    """
    Extreme cropping: Crop the box and keep on only the left leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftleg_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 22, 23, 24] + [25 + i for i in [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def full_body(keypoints_2d: np.ndarray) -> bool:
    """
    Check if all main body joints are visible.
    Args:
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """

    body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    body_keypoints = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
    return (np.maximum(keypoints_2d[body_keypoints, -1], keypoints_2d[body_keypoints_openpose, -1]) > 0).sum() == len(body_keypoints)


def upper_body(keypoints_2d: np.ndarray):
    """
    Check if all upper body joints are visible.
    Args:
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    lower_body_keypoints_openpose = [10, 11, 13, 14]
    lower_body_keypoints = [25 + i for i in [1, 0, 4, 5]]
    upper_body_keypoints_openpose = [0, 1, 15, 16, 17, 18]
    upper_body_keypoints = [25+8, 25+9, 25+12, 25+13, 25+17, 25+18]
    return ((keypoints_2d[lower_body_keypoints + lower_body_keypoints_openpose, -1] > 0).sum() == 0)\
       and ((keypoints_2d[upper_body_keypoints + upper_body_keypoints_openpose, -1] > 0).sum() >= 2)


def get_bbox(keypoints_2d: np.ndarray, rescale: float = 1.2) -> Tuple:
    """
    Get center and scale for bounding box from openpose detections.
    Args:
        keypoints_2d (np.ndarray): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center (np.ndarray): Array of shape (2,) containing the new bounding box center.
        scale (float): New bounding box scale.
    """
    valid = keypoints_2d[:,-1] > 0
    valid_keypoints = keypoints_2d[valid][:,:-1]
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
    # adjust bounding box tightness
    scale = bbox_size
    scale *= rescale
    return center, scale
