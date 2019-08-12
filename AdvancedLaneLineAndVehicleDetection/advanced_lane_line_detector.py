"""Provide implementation for line line detection."""

import cv2
import numpy as np

from abstract_image_processor import AbstractImageProcessor
from line import Line


XM_PER_PIXEL = 3.7 / 700  # meters per pixel in x dimension
YM_PER_PIXEL = 30 / 720   # meters per pixel in y dimension


class AdvancedLaneLineDetector(AbstractImageProcessor):
    """Representation of entity which detects lane lines."""
    def __init__(
            self,
            time_window: int,
            ret: int,
            mtx: int,
            dist: int,
            rvecs: int,
            tvecs: int,
            font=cv2.FONT_HERSHEY_COMPLEX):
        """Contruct ALLD object."""
        self._processed_frames = 0
        self._ret = ret
        self._mtx = mtx
        self._dist = dist
        self._rvecs = rvecs
        self._tvecs = tvecs
        self._font = font
        self._yellow_hsv_th_min = np.array([0, 70, 70])
        self._yellow_hsv_th_max = np.array([50, 255, 255])
        self._line_lt = Line(buffer_len=time_window)
        self._line_rt = Line(buffer_len=time_window)
        self._img_undistorted = None

    @property
    def img_undistorted(self):
        """Return undistorted frame."""
        return self._img_undistorted
    

    def _undistort(self, frame: np.ndarray, mtx, dist) -> np.ndarray:
        """Undistort a frame given camera matrix and distortion coefficients."""
        frame_undistorted = cv2.undistort(
            frame,
            self._mtx,
            self._dist,
            newCameraMatrix=self._mtx
        )

        return frame_undistorted

    def _thresh_frame_sobel(
            self,
            frame,
            kernel_size
            ):
        """
        Return result of sobel edge detection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

        _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)

        return sobel_mag.astype(bool)

    def _thresh_frame_in_hsv(
            self,
            frame,
            min_values,
            max_values
        ):
        """Threshold a color frame in HSV space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        min_th_ok = np.all(hsv > min_values, axis=2)
        max_th_ok = np.all(hsv < max_values, axis=2)

        out = np.logical_and(min_th_ok, max_th_ok)

        return out

    def _get_binary_from_equalized_grayscale(self, frame: np.ndarray):
        """
        Apply histogram equalization to an input frame, threshold it and return the (binary) result.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eq_global = cv2.equalizeHist(gray)

        _, th = cv2.threshold(
            eq_global,
            thresh=250,
            maxval=255,
            type=cv2.THRESH_BINARY)

        return th

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """
        Convert an input frame to a binary image which highlight as most as possible the lane-lines.
        """
        h, w = img.shape[:2]

        binary = np.zeros(shape=(h, w), dtype=np.uint8)

        hsv_yellow_mask = self._thresh_frame_in_hsv(
            img,
            self._yellow_hsv_th_min,
            self._yellow_hsv_th_max,
        )
        binary = np.logical_or(binary, hsv_yellow_mask)

        eq_white_mask = self._get_binary_from_equalized_grayscale(img)
        binary = np.logical_or(binary, eq_white_mask)

        sobel_mask = self._thresh_frame_sobel(img, kernel_size=9)
        binary = np.logical_or(binary, sobel_mask)

        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(
            binary.astype(np.uint8),
            cv2.MORPH_CLOSE, kernel
        )

        return closing

    def _birdeye(self, img: np.ndarray) -> np.ndarray:
        """Return the bird's eye view."""
        h, w = img.shape[:2]

        src = np.float32(
            [
                [w, h - 10],
                [0, h - 10],
                [546, 460],
                [732, 460]
            ]
        )
        dst = np.float32(
            [
                [w, h],
                [0, h],
                [0, 0],
                [w, 0]
            ]
        )

        m = cv2.getPerspectiveTransform(src, dst)
        minv = cv2.getPerspectiveTransform(dst, src)

        warped = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR)

        return warped, m, minv

    def _get_fits_by_previous_fits(
            self,
            birdeye_binary,
            line_lt,
            line_rt, 
            margin=100):
        """
        Get polynomial coefficients for lane-lines detected in an binary image.
        This function starts from previously detected lane-lines to speed-up the search of lane-lines in the current frame.
        """
        height, width = birdeye_binary.shape

        left_fit_pixel = line_lt.last_fit_pixel
        right_fit_pixel = line_rt.last_fit_pixel

        nonzero = birdeye_binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_lane_inds = (
            (nonzero_x > (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] - margin)) &
            (nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin))
        )
        right_lane_inds = (
            (nonzero_x > (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] - margin)) &
            (nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))

        line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
        line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

        detected = True
        if not list(line_lt.all_x) or not list(line_lt.all_y):
            left_fit_pixel = line_lt.last_fit_pixel
            left_fit_meter = line_lt.last_fit_meter
            detected = False
        else:
            left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
            left_fit_meter = np.polyfit(
                line_lt.all_y * YM_PER_PIXEL, line_lt.all_x * XM_PER_PIXEL, 2)

        if not list(line_rt.all_x) or not list(line_rt.all_y):
            right_fit_pixel = line_rt.last_fit_pixel
            right_fit_meter = line_rt.last_fit_meter
            detected = False
        else:
            right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
            right_fit_meter = np.polyfit(
                line_rt.all_y * YM_PER_PIXEL, line_rt.all_x * XM_PER_PIXEL, 2)

        line_lt.update_line(
            left_fit_pixel,
            left_fit_meter,
            detected
        )

        line_rt.update_line(
            right_fit_pixel,
            right_fit_meter,
            detected
        )

        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_fit_pixel[0] * ploty ** 2 + \
            left_fit_pixel[1] * ploty + left_fit_pixel[2]
        right_fitx = right_fit_pixel[0] * ploty ** 2 + \
            right_fit_pixel[1] * ploty + right_fit_pixel[2]

        img_fit = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255
        window_img = np.zeros_like(img_fit)

        img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        img_fit[nonzero_y[right_lane_inds],
                nonzero_x[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)

        return line_lt, line_rt, img_fit

    def _get_fits_by_sliding_windows(
            self,
            birdeye_binary,
            line_lt,
            line_rt,
            n_windows=9,
            margin=100, 
            minpin=50):
        """
        Get polynomial coefficients for lane-lines detected in an binary image.
        """
        height, width = birdeye_binary.shape

        histogram = np.sum(birdeye_binary[height//2:-30, :], axis=0)

        out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

        midpoint = len(histogram) // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(height / n_windows)

        nonzero = birdeye_binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(n_windows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(
                out_img,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (0, 255, 0), 2)
            cv2.rectangle(
                out_img,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (0, 255, 0), 2)

            good_left_inds = (
                (nonzero_y >= win_y_low) &
                (nonzero_y < win_y_high) &
                (nonzero_x >= win_xleft_low) &
                (nonzero_x < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzero_y >= win_y_low) &
                (nonzero_y < win_y_high) &
                (nonzero_x >= win_xright_low) &
                (nonzero_x < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpin:
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > minpin:
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
        line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

        detected = True
        if not list(line_lt.all_x) or not list(line_lt.all_y):
            left_fit_pixel = line_lt.last_fit_pixel
            left_fit_meter = line_lt.last_fit_meter
            detected = False
        else:
            left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
            left_fit_meter = np.polyfit(
                line_lt.all_y * YM_PER_PIXEL, line_lt.all_x * XM_PER_PIXEL, 2)

        if not list(line_rt.all_x) or not list(line_rt.all_y):
            right_fit_pixel = line_rt.last_fit_pixel
            right_fit_meter = line_rt.last_fit_meter
            detected = False
        else:
            right_fit_pixel = np.polyfit(
                line_rt.all_y,
                line_rt.all_x,
                2
            )

            right_fit_meter = np.polyfit(
                line_rt.all_y * YM_PER_PIXEL,
                line_rt.all_x * XM_PER_PIXEL, 2
            )

        line_lt.update_line(
            left_fit_pixel,
            left_fit_meter,
            detected
        )
        line_rt.update_line(
            right_fit_pixel,
            right_fit_meter,
            detected
        )
        """
        ploty = np.linspace(0, height - 1, height)
        
        left_fitx = left_fit_pixel[0] * ploty ** 2 + \
            left_fit_pixel[1] * ploty + left_fit_pixel[2]
        right_fitx = right_fit_pixel[0] * ploty ** 2 + \
            right_fit_pixel[1] * ploty + right_fit_pixel[2]
        """

        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        return line_lt, line_rt, out_img

    def _compute_offset_from_center(
            self,
            line_lt: Line,
            line_rt: Line,
            frame_width: int) -> float:
        """Compute offset from center of the inferred lane."""
        if line_lt.detected and line_rt.detected:
            line_lt_bottom = np.mean(
                line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
            line_rt_bottom = np.mean(
                line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
            lane_width = line_rt_bottom - line_lt_bottom
            midpoint = frame_width / 2
            offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
            offset_meter = XM_PER_PIXEL * offset_pix
        else:
            offset_meter = -1

        return offset_meter
    
    def _draw_back_onto_the_road(
            self,
            img_undistorted: np.ndarray,
            minv,
            line_lt: Line,
            line_rt: Line,
            keep_state: bool
            ):
        """
        Draw both the drivable lane area and the detected lane-lines onto the original (undistorted) frame.
        """
        height, width, _ = img_undistorted.shape

        left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
        right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + \
            right_fit[1] * ploty + right_fit[2]

        road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
        road_dewarped = cv2.warpPerspective(road_warp, minv, (width, height))

        blend_onto_road = cv2.addWeighted(
            img_undistorted,
            1.,
            road_dewarped,
            0.3,
            0
        )

        line_warp = np.zeros_like(img_undistorted)
        line_warp = line_lt.draw(line_warp, color=(255, 0, 0), average=keep_state)
        line_warp = line_rt.draw(line_warp, color=(0, 0, 255), average=keep_state)
        line_dewarped = cv2.warpPerspective(line_warp, minv, (width, height))

        lines_mask = blend_onto_road.copy()
        idx = np.any([line_dewarped != 0][0], axis=2)
        lines_mask[idx] = line_dewarped[idx]

        blend_onto_road = cv2.addWeighted(
            src1=lines_mask,
            alpha=0.8,
            src2=blend_onto_road,
            beta=0.5,
            gamma=0.
        )

        return blend_onto_road

    def _prepare_out_blend_frame(
            self,
            blend_on_road,
            img_binary,
            img_birdeye,
            img_fit,
            line_lt,
            line_rt,
            offset_meter):
        """
        Prepare the final p output blend, given all intermediate pipeline images.
        """
        h, w = blend_on_road.shape[:2]

        thumb_ratio = 0.2
        thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

        off_x, off_y = 20, 15

        # add a gray rectangle to highlight the upper area
        mask = blend_on_road.copy()
        mask = cv2.rectangle(
            mask,
            pt1=(0, 0),
            pt2=(w, thumb_h+2*off_y),
            color=(0, 0, 0), 
            thickness=cv2.FILLED
        )

        blend_on_road = cv2.addWeighted(
            src1=mask,
            alpha=0.2,
            src2=blend_on_road,
            beta=0.8,
            gamma=0
        )

        thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
        thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
        blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

        thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
        thumb_birdeye = np.dstack(
            [thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
        blend_on_road[off_y:thumb_h+off_y, 2*off_x + thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

        thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
        blend_on_road[off_y:thumb_h+off_y, 3*off_x+2 * thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

        mean_curvature_meter = np.mean(
            [line_lt.curvature_meter, line_rt.curvature_meter])

        cv2.putText(
            blend_on_road,
            'Curvature radius: {:.02f}m'.format(mean_curvature_meter),
            (840, 60),
            self._font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            blend_on_road,
            'Offset from center: {:.02f}m'.format(offset_meter),
            (840, 130),
            self._font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return blend_on_road

    def process(self, frame: np.ndarray, keep_state: bool=True) -> np.ndarray:
        """Extract line lanes from a given frame."""
        self._img_undistorted = self._undistort(frame, self._mtx, self._dist)

        img_binary = self._binarize(self._img_undistorted)

        img_birdeye, M, Minv = self._birdeye(img_binary)

        if (
            self._processed_frames > 0 and
            keep_state and
            self._line_lt.detected and
            self._line_rt.detected
        ):
            self._line_lt, self._line_rt, img_fit = self._get_fits_by_previous_fits(
                img_birdeye,
                self._line_lt,
                self._line_rt
            )
        else:
            line_lt, line_rt, img_fit = self._get_fits_by_sliding_windows(
                img_birdeye,
                self._line_lt,
                self._line_rt,
                n_windows=9,
            )

        offset_meter = self._compute_offset_from_center(
            self._line_lt,
            self._line_rt,
            frame.shape[1]
        )

        blend_on_road = self._draw_back_onto_the_road(
            self._img_undistorted,
            Minv,
            self._line_lt,
            self._line_rt,
            keep_state
        )

        blend_output = self._prepare_out_blend_frame(
            blend_on_road,
            img_binary,
            img_birdeye,
            img_fit,
            self._line_lt,
            self._line_rt,
            offset_meter
        )

        self._processed_frames += 1

        return blend_output