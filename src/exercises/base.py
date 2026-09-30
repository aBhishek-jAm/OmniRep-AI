"""
Base Exercise class for rule-based exercise tracking.
"""

from abc import ABC, abstractmethod


class Exercise(ABC):
    """Abstract base class for exercise trackers."""

    @abstractmethod
    def exercise(self, source):
        """
        Run the exercise tracker on a video source.

        Args:
            source: Video source (camera index or file path)
        """
        raise NotImplementedError("Subclasses must implement exercise()")
