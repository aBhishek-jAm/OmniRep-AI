"""
Lunge exercise tracker with joint angle visualization and rep counting.

Tracks bilateral knee angles with progress bars and counts repetitions
based on angle thresholds.
"""

import cv2
import numpy as np
import mediapipe as mp

from src.exercises.base import Exercise
from src.exercises.threaded_camera import ThreadedCamera
from src.utils import angle_between_lines, get_idx_to_coordinates, rescale_frame

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

POSE_LANDMARK_SPEC = mp_drawing.DrawingSpec(thickness=5, circle_radius=2, color=(0, 0, 255))
POSE_CONNECTION_SPEC = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))


class Lunges(Exercise):
    """Lunge exercise tracker with bilateral angle visualization and rep counting."""

    def exercise(self, source):
        """
        Run the lunge tracker on a video source.

        Displays:
        - Left and right knee angles with visual arcs
        - Progress bars for both legs
        - Rep counter

        Args:
            source: Video source (camera index or file path)
        """
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        threaded_camera = ThreadedCamera(source)
        count = 0
        frames = 0
        performed_lunge = False
        ang1, ang2 = 0, 0

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

            # Track knee angles
            try:
                # Left side (hip-knee-ankle: 23-25-27)
                if all(k in idx_to_coordinates for k in [23, 25, 27]):
                    self._draw_limb(image, idx_to_coordinates, 23, 25, 27)
                    ang1 = angle_between_lines(
                        (idx_to_coordinates[23], idx_to_coordinates[25]),
                        (idx_to_coordinates[25], idx_to_coordinates[27])
                    )
                    cv2.putText(image, str(round(ang1, 2)),
                                (idx_to_coordinates[25][0] - 40, idx_to_coordinates[25][1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                # Right side (hip-knee-ankle: 24-26-28)
                if all(k in idx_to_coordinates for k in [24, 26, 28]):
                    self._draw_limb(image, idx_to_coordinates, 24, 26, 28)
                    ang2 = angle_between_lines(
                        (idx_to_coordinates[24], idx_to_coordinates[26]),
                        (idx_to_coordinates[26], idx_to_coordinates[28])
                    )
                    cv2.putText(image, str(round(ang2, 2)),
                                (idx_to_coordinates[26][0] - 40, idx_to_coordinates[26][1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            except Exception:
                pass

            # Count reps and draw progress bars
            try:
                frames += 1
                if frames > 80:  # Skip initial frames for stability
                    if ang1 < 100:
                        performed_lunge = True
                    if ang1 > 150 and performed_lunge:
                        count += 1
                        performed_lunge = False

                # Left leg progress bar
                display_ang1 = 180 - ang1
                c1 = (0, 255, 0) if display_ang1 > 70 else (255, 0, 0)
                h, w = image.shape[:2]
                bar_x = min(550, w - 70)
                bar = int(np.interp(display_ang1, (10, 70), (min(850, h - 50), 300)))
                per = int(np.interp(display_ang1, (10, 70), (0, 100)))
                cv2.rectangle(image, (bar_x, 300), (bar_x + 60, min(850, h - 50)), c1)
                cv2.rectangle(image, (bar_x, bar), (bar_x + 60, min(850, h - 50)), c1, cv2.FILLED)
                cv2.putText(image, f'{per}%', (bar_x, 255),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, c1, 4)

                # Right leg progress bar
                display_ang2 = 180 - ang2
                c2 = (0, 255, 0) if display_ang2 > 40 else (255, 0, 0)
                bar_x2 = min(w - 70, 1400)
                bar2 = int(np.interp(display_ang2, (10, 40), (min(850, h - 50), 300)))
                per2 = int(np.interp(display_ang2, (10, 40), (0, 100)))
                cv2.rectangle(image, (bar_x2, 300), (bar_x2 + 60, min(850, h - 50)), c2)
                cv2.rectangle(image, (bar_x2, bar2), (bar_x2 + 60, min(850, h - 50)), c2, cv2.FILLED)
                cv2.putText(image, f'{per2}%', (bar_x2, 255),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, c2, 4)
            except Exception:
                pass

            if 0 in idx_to_coordinates:
                cv2.putText(image, f"Lunges: {count}",
                            (idx_to_coordinates[0][0] - 40, idx_to_coordinates[0][1] + 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 5)

            cv2.imshow('FitVision - Lunge Tracker', rescale_frame(image, percent=100))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        pose.close()
        threaded_camera.release()

    @staticmethod
    def _draw_limb(image, coords, idx1, idx2, idx3):
        """Draw limb lines and joint circles."""
        cv2.line(image, coords[idx1], coords[idx2], thickness=6, color=(255, 0, 0))
        cv2.line(image, coords[idx2], coords[idx3], thickness=6, color=(255, 0, 0))
        for idx in [idx1, idx2, idx3]:
            cv2.circle(image, coords[idx], 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, coords[idx], 15, (0, 0, 255), 2)
