"""
Squat exercise tracker with joint angle visualization and rep counting.

Tracks knee angles, elbow angles, and back angles to monitor squat form,
and counts repetitions based on hip-to-knee vertical distance.
"""

import cv2
import numpy as np
import mediapipe as mp

from src.exercises.base import Exercise
from src.exercises.threaded_camera import ThreadedCamera
from src.utils import (
    angle_between_lines, get_idx_to_coordinates,
    convert_arc, draw_ellipse, rescale_frame
)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

POSE_LANDMARK_SPEC = mp_drawing.DrawingSpec(thickness=5, circle_radius=2, color=(0, 0, 255))
POSE_CONNECTION_SPEC = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))


class Squat(Exercise):
    """Squat exercise tracker with angle visualization and rep counting."""

    def exercise(self, source):
        """
        Run the squat tracker on a video source.

        Displays:
        - Knee angles (both sides)
        - Elbow angles (both sides)
        - Back/hip angles (both sides)
        - Rep counter

        Args:
            source: Video source (camera index or file path)
        """
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        threaded_camera = ThreadedCamera(source)
        count = 0
        performed_squat = False

        while True:
            success, image = threaded_camera.show_frame()
            if not success or image is None:
                continue

            image = cv2.flip(image, 1)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=POSE_LANDMARK_SPEC,
                connection_drawing_spec=POSE_CONNECTION_SPEC
            )

            idx_to_coordinates = get_idx_to_coordinates(image, results)

            # Knee angle visualization
            try:
                self._draw_knee_angles(image, idx_to_coordinates)
            except Exception:
                pass

            # Elbow angle visualization
            try:
                self._draw_elbow_angles(image, idx_to_coordinates)
            except Exception:
                pass

            # Back angle visualization
            try:
                self._draw_back_angles(image, idx_to_coordinates)
            except Exception:
                pass

            # Count squats
            try:
                hip_coord = idx_to_coordinates.get(24, idx_to_coordinates.get(23))
                knee_coord = idx_to_coordinates.get(26, idx_to_coordinates.get(25))

                if hip_coord and knee_coord:
                    if abs(hip_coord[1] - knee_coord[1]) < 35:
                        performed_squat = True
                    if abs(hip_coord[1] - knee_coord[1]) > 35 and performed_squat:
                        count += 1
                        performed_squat = False
            except Exception:
                pass

            if 0 in idx_to_coordinates:
                cv2.putText(image, f"Squats: {count}",
                            (idx_to_coordinates[0][0] - 30, idx_to_coordinates[0][1] - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            cv2.imshow('FitVision - Squat Tracker', rescale_frame(image, percent=100))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        pose.close()
        threaded_camera.release()

    def _draw_knee_angles(self, image, coords):
        """Draw knee angle arcs for both sides."""
        # Right side
        if all(k in coords for k in [24, 26, 28]):
            self._draw_angle_arc(image, coords[24], coords[26], coords[28])
        # Left side
        if all(k in coords for k in [23, 25, 27]):
            self._draw_angle_arc(image, coords[23], coords[25], coords[27])

    def _draw_elbow_angles(self, image, coords):
        """Draw elbow angle arcs for both sides."""
        # Right side
        if all(k in coords for k in [12, 14, 16]):
            self._draw_angle_arc(image, coords[12], coords[14], coords[16])
        # Left side
        if all(k in coords for k in [11, 13, 15]):
            self._draw_angle_arc(image, coords[11], coords[13], coords[15])

    def _draw_back_angles(self, image, coords):
        """Draw back/hip angle arcs."""
        if all(k in coords for k in [12, 24, 26]):
            self._draw_angle_arc(image, coords[12], coords[24], coords[26])
        elif all(k in coords for k in [11, 23, 25]):
            self._draw_angle_arc(image, coords[11], coords[23], coords[25])

    @staticmethod
    def _draw_angle_arc(image, pt1, pt2, pt3):
        """Draw angle visualization between three points."""
        l1 = np.linspace(pt1, pt2, 100)
        l2 = np.linspace(pt2, pt3, 100)

        cv2.line(image, (int(l1[99][0]), int(l1[99][1])),
                 (int(l1[69][0]), int(l1[69][1])), thickness=4, color=(0, 0, 255))
        cv2.line(image, (int(l2[0][0]), int(l2[0][1])),
                 (int(l2[30][0]), int(l2[30][1])), thickness=4, color=(0, 0, 255))

        ang = angle_between_lines((pt1, pt2), (pt2, pt3))
        cv2.putText(image, str(round(ang, 2)), pt2,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        try:
            center, radius, start_angle, end_angle = convert_arc(l1[90], l2[10], sagitta=15)
            axes = (radius, radius)
            draw_ellipse(image, center, axes, -1, start_angle, end_angle, 255)
        except Exception:
            pass
