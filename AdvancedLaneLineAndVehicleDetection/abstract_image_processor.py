"""Provide a basic abstraction for base Image processor class."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractImageProcessor(ABC):
    """Representation of an abstract image processor."""

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Perform image processing."""
        pass
