"""
Pushup exercise tracker with joint angle visualization and rep counting.

Tracks shoulder-ankle-wrist angles, elbow angles, and wrist angles
to monitor pushup form and count repetitions.
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


class Pushup(Exercise):
    """Pushup exercise tracker with angle visualization and rep counting."""

    def exercise(self, source):
        """
        Run the pushup tracker on a video source.

        Displays:
        - Shoulder-ankle-wrist angle
        - Shoulder-elbow-wrist angle
        - Elbow-wrist-ground angle
        - Rep counter

        Args:
            source: Video source (camera index or file path)
        """
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        threaded_camera = ThreadedCamera(source)
        count = 0
        performed_pushup = False

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

            # Shoulder-ankle-wrist angle
            try:
                if all(k in idx_to_coordinates for k in [12, 28, 16]):
                    self._draw_angle(image, idx_to_coordinates, 12, 28, 16)
                elif all(k in idx_to_coordinates for k in [11, 27, 15]):
                    self._draw_angle(image, idx_to_coordinates, 11, 27, 15)
            except Exception:
                pass

            # Shoulder-elbow-wrist angle
            try:
                if all(k in idx_to_coordinates for k in [12, 14, 16]):
                    self._draw_angle(image, idx_to_coordinates, 12, 14, 16)
                elif all(k in idx_to_coordinates for k in [11, 13, 15]):
                    self._draw_angle(image, idx_to_coordinates, 11, 13, 15)
            except Exception:
                pass

            # Elbow-wrist-ground angle
            try:
                if all(k in idx_to_coordinates for k in [14, 16]):
                    wrist = idx_to_coordinates[16]
                    ground_pt = (wrist[0] + 80, wrist[1])
                    self._draw_angle_with_ground(image, idx_to_coordinates[14], wrist, ground_pt)
                elif all(k in idx_to_coordinates for k in [13, 15]):
                    wrist = idx_to_coordinates[15]
                    ground_pt = (wrist[0] + 80, wrist[1])
                    self._draw_angle_with_ground(image, idx_to_coordinates[13], wrist, ground_pt)
            except Exception:
                pass

            # Count pushups
            try:
                shoulder = idx_to_coordinates.get(12, idx_to_coordinates.get(11))
                wrist = idx_to_coordinates.get(16, idx_to_coordinates.get(15))

                if shoulder and wrist:
                    if abs(shoulder[1] - wrist[1]) < 300:
                        performed_pushup = True
                    if abs(shoulder[1] - wrist[1]) > 300 and performed_pushup:
                        count += 1
                        performed_pushup = False
            except Exception:
                pass

            if 0 in idx_to_coordinates:
                cv2.putText(image, f"Pushups: {count}",
                            (idx_to_coordinates[0][0] - 60, idx_to_coordinates[0][1] - 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            cv2.imshow('FitVision - Pushup Tracker', rescale_frame(image, percent=100))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        pose.close()
        threaded_camera.release()

    @staticmethod
    def _draw_angle(image, coords, idx1, idx2, idx3):
        """Draw angle visualization between three indexed landmarks."""
        pt1, pt2, pt3 = coords[idx1], coords[idx2], coords[idx3]
        cv2.line(image, pt1, pt2, thickness=4, color=(255, 0, 255))
        cv2.line(image, pt2, pt3, thickness=4, color=(255, 0, 255))

        l1 = np.linspace(pt1, pt2, 100)
        l2 = np.linspace(pt2, pt3, 100)

        ang = angle_between_lines((pt1, pt2), (pt2, pt3))
        cv2.putText(image, f"  {round(ang, 2)}", pt2,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        try:
            center, radius, start_angle, end_angle = convert_arc(l1[80], l2[20], sagitta=15)
            draw_ellipse(image, center, (radius, radius), -1, start_angle, end_angle, 255)
        except Exception:
            pass

    @staticmethod
    def _draw_angle_with_ground(image, elbow_pt, wrist_pt, ground_pt):
        """Draw angle between elbow-wrist and horizontal ground."""
        cv2.line(image, elbow_pt, wrist_pt, thickness=4, color=(255, 0, 255))
        cv2.line(image, wrist_pt, ground_pt, thickness=4, color=(255, 0, 255))

        l1 = np.linspace(elbow_pt, wrist_pt, 100)
        l2 = np.linspace(wrist_pt, ground_pt, 100)

        ang = angle_between_lines((elbow_pt, wrist_pt), (wrist_pt, ground_pt))
        cv2.putText(image, f"  {round(ang, 2)}", wrist_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        try:
            center, radius, start_angle, end_angle = convert_arc(l1[80], l2[20], sagitta=15)
            draw_ellipse(image, center, (radius, radius), -1, start_angle, end_angle, 255)
        except Exception:
            pass
