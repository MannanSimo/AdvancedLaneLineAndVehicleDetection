"""Provide abtsraction for Lane Line object."""

import collections

import cv2
import numpy as np


class Line:
    """Representation of Lane Line object."""
    Y_EVAL = 0

    def __init__(self, buffer_len=10):
        """Contruct Line object."""
        # flag to mark if the line was detected the last iteration
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        self.last_fit_pixel = None
        self.last_fit_meter = None

        # list of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len)
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len)

        self.radius_of_curvature = None

        # store all pixels coords (x, y) of line detected
        self.all_x = None
        self.all_y = None

    def update_line(
            self,
            new_fit_pixel,
            new_fit_meter,
            detected,
            clear_buffer=False):
        """Update Line with new fitted coefficients."""
        self.detected = detected

        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter

        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)

    def draw(self, mask, color=(255, 0, 0), line_width=50, average=False):
        """Draw the line on a color mask image."""
        h, w, c = mask.shape

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel

        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.vstack([pts_left, pts_right])

        return cv2.fillPoly(mask, [np.int32(pts)], color)

    @property
    def average_fit(self):
        """
        Return average of polynomial coefficients of the last N iterations.
        """
        return np.mean(self.recent_fits_pixel, axis=0)

    def _calculate_line_curvature(self, coeffs):
        """Calculate radius of curvature of the line."""
        return (
            (
                (1 + (2 * coeffs[0] * self.Y_EVAL + coeffs[1]) ** 2) ** 1.5
            ) / np.absolute(2 * coeffs[0])
        )

    @property
    def curvature(self):
        """Return radius of curvature of the line (averaged)."""
        return self._calculate_line_curvature(self.average_fit)

    @property
    # radius of curvature of the line (averaged)
    def curvature_meter(self):
        return self._calculate_line_curvature(np.mean(self.recent_fits_meter, axis=0))