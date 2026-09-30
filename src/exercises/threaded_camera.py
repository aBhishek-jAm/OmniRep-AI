"""
Threaded camera capture for smoother video processing.

Uses a background thread to continuously capture frames, reducing latency
in the main processing loop.
"""

import cv2
import time
from threading import Thread


class ThreadedCamera:
    """
    Threaded camera capture that reads frames in a background thread.

    This decouples frame capture from frame processing, providing
    smoother video even when processing is slow.

    Args:
        src: Video source (camera index or file path)
        target_fps: Target frames per second (default: 60)
    """

    def __init__(self, src=0, target_fps=60):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.fps_interval = 1 / target_fps
        self.frame = None
        self.status = False

        # Start background frame capture thread
        self.thread = Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Continuously capture frames in the background."""
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(self.fps_interval)

    def show_frame(self):
        """
        Get the latest captured frame.

        Returns:
            tuple: (success_bool, frame_or_None)
        """
        return self.status, self.frame

    def release(self):
        """Release the video capture resource."""
        self.capture.release()
