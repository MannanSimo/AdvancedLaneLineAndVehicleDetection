"""Provide entity to scan a frame for vehicles."""

import glob
import os
import pickle

import cv2
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from keras.models import Model, Sequential
import numpy as np
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split as trainTestSplit
from sklearn.utils import shuffle


class VehicleScanner:
    """Representation of Vehicle objects scanner."""

    def __init__(
        self,
        img_input_shape=(720, 1280, 3),
        crop=(400, 660),
        point_size=64,
        confidence_thrd=.7,
        ve_hi_depth=30,
        group_thrd=10,
        group_diff=.1
    ):
        """Return VehicleScanner object."""
        self.crop = crop
        self.detection_point_size = point_size
        self.confidence_thrd = confidence_thrd

        bottom_clip = img_input_shape[0] - crop[1]
        in_h = img_input_shape[0] - crop[0] - bottom_clip
        in_w = img_input_shape[1]
        in_ch = img_input_shape[2]

        self.cnn_model, cnn_model_name = self._get_model(
            input_shape=(in_h, in_w, in_ch))

        self.cnn_model.load_weights('../model_data/{}.h5'.format(cnn_model_name))

        self.ve_hi_depth = ve_hi_depth
        self.group_thrd = group_thrd
        self.group_diff = group_diff
        self._vehicle_boxes_history = []

        self.diag_kernel = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]

    def _get_model(self, input_shape=(64, 64, 3)):
        """Return Keras model and model's name for FCNN."""
        model = Sequential()

        model.add(
            Lambda(
                lambda x: x / 255., input_shape=input_shape, output_shape=input_shape
            )
        )

        model.add(
            Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation='relu',
                name='cv0',
                input_shape=input_shape,
                padding="same"
            )
        )
        model.add(Dropout(0.5))

        model.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation='relu',
                name='cv1',
                padding="same"
            )
        )
        model.add(Dropout(0.5))

        model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation='relu',
                name='cv2',
                padding="same"
            )
        )
        model.add(MaxPooling2D(pool_size=(8, 8)))
        model.add(Dropout(0.5))

        model.add(
            Conv2D(
                filters=1,
                kernel_size=(8, 8),
                name='fcn',
                activation="sigmoid"
            )
        )

        return model, 'ppico'

    def _search_vehicle_objects(self, img):
        """Return points of potential vehicle(s) location."""
        # Cropping to the region of interest
        roi = img[self.crop[0]:self.crop[1], :]

        roi_w, roi_h = roi.shape[1], roi.shape[0]

        roi = np.expand_dims(roi, axis=0)

        detection_map = self.cnn_model.predict(roi)

        prediction_map_h, prediction_map_w = detection_map.shape[1], detection_map.shape[2]

        ratio_h, ratio_w = roi_h / prediction_map_h, roi_w / prediction_map_w

        detection_map = detection_map.reshape(
            detection_map.shape[1],
            detection_map.shape[2]
        )

        detection_map = detection_map > self.confidence_thrd

        labels = label(detection_map, structure=self.diag_kernel)

        hot_points = []

        # Considering obtained labels as vehicles
        for vehicle_id in range(labels[1]):
            nz = (labels[0] == vehicle_id + 1).nonzero()
            nz_y = np.array(nz[0])
            nz_x = np.array(nz[1])

            x_min = np.min(nz_x) - 32
            x_max = np.max(nz_x) + 32

            y_min = np.min(nz_y)
            y_max = np.max(nz_y) + 64

            span_x = x_max - x_min
            span_y = y_max - y_min

            for x, y in zip(nz_x, nz_y):
                offset_x = (x - x_min) / span_x * self.detection_point_size
                offset_y = (y - y_min) / span_y * self.detection_point_size

                # ROI - region of interest
                # Getting boundaries in ROI coordinates scale (multiplying by ratioW, ratioH)
                top_left_x = int(round(x * ratio_w - offset_x, 0))
                top_left_y = int(round(y * ratio_h - offset_y, 0))
                bottom_left_x = top_left_x + self.detection_point_size
                bottom_left_y = top_left_y + self.detection_point_size

                top_left = (top_left_x, self.crop[0] + top_left_y)
                bottom_right = (bottom_left_x, self.crop[0] + bottom_left_y)

                hot_points.append((top_left, bottom_right))

        return hot_points

    def _add_heat(self, mask, bounding_boxes):
        """Creates the actual heat map."""
        for box in bounding_boxes:
            top_y = box[0][1]
            bottom_y = box[1][1]
            left_x = box[0][0]
            right_x = box[1][0]

            mask[top_y:bottom_y, left_x:right_x] += 1

            mask = np.clip(mask, 0, 255)

        return mask

    def _get_hot_regions(self, src):
        """Return hot points and heat map."""
        hot_points = self._search_vehicle_objects(src)
        sample_mask = np.zeros_like(src[:, :, 0]).astype(np.float)
        heat_map = self._add_heat(sample_mask, hot_points)

        current_frame_boxes = label(heat_map, structure=self.diag_kernel)

        return current_frame_boxes, heat_map

    def _update_history(self, current_labels):
        """Converting hot regions to bounding boxes and saving them to boxes history list."""
        for i in range(current_labels[1]):
            nz = (current_labels[0] == i + 1).nonzero()
            nz_y = np.array(nz[0])
            nz_x = np.array(nz[1])

            tl_x = np.min(nz_x)
            tl_y = np.min(nz_y)
            br_x = np.max(nz_x)
            br_y = np.max(nz_y)

            self._vehicle_boxes_history.append([tl_x, tl_y, br_x, br_y])

            self._vehicle_boxes_history = self._vehicle_boxes_history[-self.ve_hi_depth:]

    def get_boxes(self, src):
        """Extract vehicles bounding boxes."""
        current_labels, _ = self._get_hot_regions(src)

        self._update_history(current_labels)

        boxes, _ = cv2.groupRectangles(
            rectList=np.array(self._vehicle_boxes_history).tolist(),
            groupThreshold=self.group_thrd, eps=self.group_diff
        )

        return boxes
