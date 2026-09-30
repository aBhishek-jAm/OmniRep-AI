"""
Shared utility functions for pose estimation, angle calculation, and visualization.
"""

import math
import cv2
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b formed by the line segments a-b and b-c.

    Args:
        a: First point [x, y] or [x, y, z]
        b: Middle point (vertex) [x, y] or [x, y, z]
        c: Third point [x, y] or [x, y, z]

    Returns:
        float: Angle in degrees (0-180)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def get_landmark_coords(landmarks, index):
    """
    Get (x, y) coordinates of a specific landmark.

    Args:
        landmarks: List of pose landmarks from MediaPipe
        index: Landmark index

    Returns:
        list: [x, y] coordinates
    """
    return [landmarks[index].x, landmarks[index].y]


def get_landmark_coords_3d(landmarks, index):
    """
    Get (x, y, z, visibility) of a specific landmark.

    Args:
        landmarks: List of pose landmarks from MediaPipe
        index: Landmark index

    Returns:
        list: [x, y, z, visibility]
    """
    lm = landmarks[index]
    return [lm.x, lm.y, lm.z, lm.visibility]


def extract_basic_landmarks(results):
    """
    Extract all 33 pose landmarks as a flat array of [x, y, z, visibility] values.

    Args:
        results: MediaPipe pose processing results

    Returns:
        list: Flattened landmark values (132 values for 33 landmarks × 4)
    """
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    return landmarks


def extract_enhanced_landmarks(results):
    """
    Extract pose landmarks with additional engineered features:
    - 33 landmarks × 4 values = 132 base features
    - 2 distance features (shoulder width, torso length)
    - 4 angle features (left/right elbow, left/right knee)
    Total: 138 features

    Args:
        results: MediaPipe pose processing results

    Returns:
        list: Flattened feature values (138 values)
    """
    landmarks = []
    if results.pose_landmarks:
        lm_list = results.pose_landmarks.landmark

        # Basic landmarks (132 features)
        for lm in lm_list:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

        # Additional engineered features (6 features)
        if len(lm_list) >= 32:
            landmarks.extend(calculate_key_distances(lm_list))
            landmarks.extend(calculate_key_angles(lm_list))

    return landmarks


def calculate_key_distances(landmarks):
    """
    Calculate normalized distances between important body points.

    Args:
        landmarks: List of MediaPipe landmark objects

    Returns:
        list: [shoulder_width, torso_length]
    """
    distances = []

    # Shoulder width
    left_shoulder = [landmarks[11].x, landmarks[11].y]
    right_shoulder = [landmarks[12].x, landmarks[12].y]
    shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
    distances.append(shoulder_width)

    # Torso length (left shoulder to left hip)
    torso_length = np.linalg.norm(
        np.array([landmarks[11].x, landmarks[11].y]) -
        np.array([landmarks[23].x, landmarks[23].y])
    )
    distances.append(torso_length)

    return distances


def calculate_key_angles(landmarks):
    """
    Calculate important joint angles for exercise classification.

    Args:
        landmarks: List of MediaPipe landmark objects

    Returns:
        list: [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]
    """
    angles = []

    # Left elbow angle
    angles.append(calculate_angle(
        [landmarks[11].x, landmarks[11].y],
        [landmarks[13].x, landmarks[13].y],
        [landmarks[15].x, landmarks[15].y]
    ))

    # Right elbow angle
    angles.append(calculate_angle(
        [landmarks[12].x, landmarks[12].y],
        [landmarks[14].x, landmarks[14].y],
        [landmarks[16].x, landmarks[16].y]
    ))

    # Left knee angle
    angles.append(calculate_angle(
        [landmarks[23].x, landmarks[23].y],
        [landmarks[25].x, landmarks[25].y],
        [landmarks[27].x, landmarks[27].y]
    ))

    # Right knee angle
    angles.append(calculate_angle(
        [landmarks[24].x, landmarks[24].y],
        [landmarks[26].x, landmarks[26].y],
        [landmarks[28].x, landmarks[28].y]
    ))

    return angles


def get_mediapipe_coordinates(landmarks, mp_pose, side, joint):
    """
    Retrieve x and y coordinates of a particular keypoint from the pose estimation model.

    Args:
        landmarks: Processed keypoints from the pose estimation model
        mp_pose: MediaPipe pose module
        side: 'left' or 'right'
        joint: 'shoulder', 'elbow', 'wrist', 'hip', 'knee', or 'ankle'

    Returns:
        list: [x, y] coordinates
    """
    coord = getattr(mp_pose.PoseLandmark, side.upper() + "_" + joint.upper())
    return [landmarks[coord.value].x, landmarks[coord.value].y]


def draw_angle_on_image(image, angle, joint_coords, frame_size=(640, 480)):
    """
    Display the joint angle value near the joint within the image frame.

    Args:
        image: OpenCV image
        angle: Angle value to display
        joint_coords: Normalized [x, y] coordinates of the joint
        frame_size: Tuple of (width, height)
    """
    cv2.putText(
        image, str(int(angle)),
        tuple(np.multiply(joint_coords, list(frame_size)).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
    )


def draw_probability_bars(image, probabilities, labels, colors=None):
    """
    Display model prediction probability distribution as horizontal bar graph.

    Args:
        image: OpenCV image
        probabilities: Array of prediction probabilities
        labels: List of class labels
        colors: Optional list of BGR color tuples
    """
    if colors is None:
        colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),
                  (245, 16, 117), (16, 245, 245)]

    output = image.copy()
    for num, prob in enumerate(probabilities):
        color = colors[num % len(colors)]
        cv2.rectangle(output, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(output, labels[num], (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def rescale_frame(frame, percent=75):
    """
    Rescale a video frame by a given percentage.

    Args:
        frame: OpenCV image
        percent: Scale percentage (default 75)

    Returns:
        Rescaled frame
    """
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_ellipse(img, center, axes, angle, startAngle, endAngle, color,
                 thickness=2, lineType=cv2.LINE_AA, shift=10):
    """
    Draw an ellipse (arc) with sub-pixel resolution.

    Args:
        img: OpenCV image
        center: Center point (x, y)
        axes: Axes lengths (a, b)
        angle: Rotation angle
        startAngle: Starting angle of the arc
        endAngle: Ending angle of the arc
        color: Color
        thickness: Line thickness
        lineType: Line type
        shift: Bit shift for sub-pixel precision
    """
    center = (
        int(round(center[0] * 2 ** shift)),
        int(round(center[1] * 2 ** shift))
    )
    axes = (
        int(round(axes[0] * 2 ** shift)),
        int(round(axes[1] * 2 ** shift))
    )
    return cv2.ellipse(img, center, axes, angle,
                       startAngle, endAngle, color,
                       thickness, lineType, shift)


def convert_arc(pt1, pt2, sagitta):
    """
    Convert two points and a sagitta into arc parameters (center, radius, angles).

    Args:
        pt1: First point (x, y)
        pt2: Second point (x, y)
        sagitta: Arc height

    Returns:
        tuple: (center, radius, pt1_angle, pt2_angle)
    """
    x1, y1 = pt1
    x2, y2 = pt2

    n = np.array([y2 - y1, x1 - x2])
    n_dist = np.sqrt(np.sum(n ** 2))

    if np.isclose(n_dist, 0):
        print('Warning: The distance between pt1 and pt2 is too small.')
        return (0, 0), 1, 0, 0

    n = n / n_dist
    x3, y3 = (np.array(pt1) + np.array(pt2)) / 2 + sagitta * n

    A = np.array([
        [x1 ** 2 + y1 ** 2, x1, y1, 1],
        [x2 ** 2 + y2 ** 2, x2, y2, 1],
        [x3 ** 2 + y3 ** 2, x3, y3, 1]
    ])
    M11 = np.linalg.det(A[:, (1, 2, 3)])
    M12 = np.linalg.det(A[:, (0, 2, 3)])
    M13 = np.linalg.det(A[:, (0, 1, 3)])
    M14 = np.linalg.det(A[:, (0, 1, 2)])

    if np.isclose(M11, 0):
        print('Warning: The third point is collinear.')
        return (0, 0), 1, 0, 0

    cx = 0.5 * M12 / M11
    cy = -0.5 * M13 / M11
    radius = np.sqrt(cx ** 2 + cy ** 2 + M14 / M11)

    pt1_angle = 180 * np.arctan2(y1 - cy, x1 - cx) / np.pi
    pt2_angle = 180 * np.arctan2(y2 - cy, x2 - cx) / np.pi

    return (cx, cy), radius, pt1_angle, pt2_angle


def dot(vA, vB):
    """Dot product of two 2D vectors."""
    return vA[0] * vB[0] + vA[1] * vB[1]


def angle_between_lines(lineA, lineB):
    """
    Calculate angle between two lines in degrees.

    Args:
        lineA: ((x1, y1), (x2, y2))
        lineB: ((x1, y1), (x2, y2))

    Returns:
        float: Angle in degrees
    """
    vA = [(lineA[0][0] - lineA[1][0]), (lineA[0][1] - lineA[1][1])]
    vB = [(lineB[0][0] - lineB[1][0]), (lineB[0][1] - lineB[1][1])]
    dot_prod = dot(vA, vB)
    magA = dot(vA, vA) ** 0.5
    magB = dot(vB, vB) ** 0.5
    angle = math.acos(max(-1, min(1, dot_prod / magA / magB)))
    ang_deg = math.degrees(angle) % 360
    ang_deg = 180 - ang_deg
    if ang_deg - 180 >= 0:
        return 360 - ang_deg
    return ang_deg


def get_idx_to_coordinates(image, results, visibility_threshold=0.5, presence_threshold=0.5):
    """
    Convert MediaPipe landmarks to pixel coordinates dictionary.

    Args:
        image: OpenCV image (for dimensions)
        results: MediaPipe pose results
        visibility_threshold: Minimum visibility
        presence_threshold: Minimum presence

    Returns:
        dict: {landmark_index: (pixel_x, pixel_y)}
    """
    from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

    idx_to_coordinates = {}
    image_rows, image_cols, _ = image.shape
    try:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if ((landmark.HasField('visibility') and
                 landmark.visibility < visibility_threshold) or
                    (landmark.HasField('presence') and
                     landmark.presence < presence_threshold)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(
                landmark.x, landmark.y, image_cols, image_rows
            )
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
    except Exception:
        pass
    return idx_to_coordinates
