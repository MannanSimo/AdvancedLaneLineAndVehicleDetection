"""Provide entity to detect vehicle(s) on a frame."""

import numpy as np
import cv2

from abstract_image_processor import AbstractImageProcessor
from vehicle_scanner import VehicleScanner


class VehicleDetector(AbstractImageProcessor):
    """Representation of entity which detects cars and draws boxes."""

    def __init__(
            self,
            ve_hi_depth=30,
            point_size=64,
            group_thrd=10,
            group_diff=.1,
            confidence_thrd=.7):
        """Return Detector object."""
        self.scanner = VehicleScanner(
            point_size=point_size,
            ve_hi_depth=ve_hi_depth,
            group_thrd=group_thrd,
            group_diff=group_diff,
            confidence_thrd=confidence_thrd
        )

    def _draw_boxes(self, img, bounding_boxes, color=(0, 255, 0), thickness=4):
        """Draw bounding boxes on the given image."""
        for b_box in bounding_boxes:

            b_box = np.array(b_box)
            b_box = b_box.reshape(b_box.size)

            cv2.rectangle(
                img=img,
                pt1=(b_box[0], b_box[1]),
                pt2=(b_box[2], b_box[3]),
                color=color,
                thickness=thickness
            )

    def process(self, undistorted_frame: np.array, target_frame: np.array) -> np.array:
        """Add detected vehicles to the original frame."""
        vehicle_boxes = self.scanner.get_boxes(undistorted_frame)

        self._draw_boxes(target_frame, vehicle_boxes)

        return target_frame
