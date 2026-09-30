"""
Plank exercise tracker with hold timer and form feedback.

Tracks shoulder-hip-ankle angle to monitor plank form and measures
total hold duration.
"""

import cv2
import numpy as np
import time
import mediapipe as mp

from src.exercises.base import Exercise
from src.exercises.threaded_camera import ThreadedCamera
from src.utils import angle_between_lines, get_idx_to_coordinates, rescale_frame

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

POSE_LANDMARK_SPEC = mp_drawing.DrawingSpec(thickness=5, circle_radius=2, color=(0, 0, 255))
POSE_CONNECTION_SPEC = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))


class Plank(Exercise):
    """Plank exercise tracker with timer and form quality feedback."""

    def exercise(self, source):
        """
        Run the plank tracker on a video source.

        Displays:
        - Shoulder-hip-ankle angle
        - Form quality progress bar
        - Total plank hold timer

        Args:
            source: Video source (camera index or file path)
        """
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        threaded_camera = ThreadedCamera(source)
        eang1 = 0
        plank_timer = None
        plank_duration = 0

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

            # Track shoulder-hip-ankle angle
            try:
                if all(k in idx_to_coordinates for k in [11, 23, 27]):
                    pts = [idx_to_coordinates[11], idx_to_coordinates[23], idx_to_coordinates[27]]
                    cv2.line(image, pts[0], pts[1], thickness=6, color=(255, 0, 0))
                    cv2.line(image, pts[1], pts[2], thickness=6, color=(255, 0, 0))

                    eang1 = angle_between_lines((pts[0], pts[1]), (pts[1], pts[2]))
                    cv2.putText(image, str(round(eang1, 2)),
                                (pts[1][0] - 40, pts[1][1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

                    for pt in pts:
                        cv2.circle(image, pt, 10, (0, 0, 255), cv2.FILLED)
                        cv2.circle(image, pt, 15, (0, 0, 255), 2)
            except Exception:
                pass

            # Timer and progress bar
            try:
                if eang1 > 170:  # Good plank form
                    if plank_timer is None:
                        plank_timer = time.time()
                    plank_duration += time.time() - plank_timer
                    plank_timer = time.time()
                else:
                    plank_timer = None

                h = image.shape[0]
                bar = int(np.interp(eang1, (120, 170), (min(850, h - 50), 300)))
                per = int(np.interp(eang1, (120, 170), (0, 100)))
                cv2.rectangle(image, (200, 300), (260, min(850, h - 50)), (0, 255, 0))
                cv2.rectangle(image, (200, bar), (260, min(850, h - 50)), (0, 255, 0), cv2.FILLED)
                cv2.putText(image, f'{per}%', (200, 255),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 4)
            except Exception:
                pass

            if 0 in idx_to_coordinates:
                cv2.putText(image, f"Plank: {round(plank_duration)}s",
                            (idx_to_coordinates[0][0] - 60, idx_to_coordinates[0][1] - 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)

            cv2.imshow('FitVision - Plank Tracker', rescale_frame(image, percent=100))
            if cv2.waitKey(5) & 0xFF == 27:
                break

        pose.close()
        threaded_camera.release()
