import copy

import cv2
import numpy as np
from matplotlib import pyplot as plt

from behaviors.Tools.functions import get_points_from_segment


class Map:

    def __init__(self, map_size, obs_range, discretization=10):

        self.segment_size = map_size * 2  # Size of the map in m
        self.obs_range = obs_range * discretization  # Observation range (m) to pixel
        self.discretization = discretization  # Number of pixel per m
        self.discret_segment_size = self.segment_size * self.discretization

        self.x_max = self.discret_segment_size
        self.y_max = self.discret_segment_size
        self.x_min = 0
        self.y_min = 0

        self.map = np.ones(self.discret_segment_size * self.discret_segment_size)
        self.map.resize((self.discret_segment_size, self.discret_segment_size))

    def get_segment_size(self):
        return self.segment_size

    def get_discret_segment_size(self):
        return self.discret_segment_size

    def get_nbr_cells(self):
        return self.map.size

    def observe_and_check_cells(self, x, y, update_idleness=False):
        """

        :param x: coord x
        :param y: coord y
        :param update_idleness: True if we just evaluate without updating the idleness
        :return: sum of idleness (negative form)
        """

        f = 0

        x_check, y_check = int(x * self.discretization), int(y * self.discretization)
        # Count the number of ones in the interval

        # Verify interval
        x_min_check = x_check - self.obs_range
        x_min_check = x_min_check if x_min_check > 0 else 0
        x_max_check = x_check + self.obs_range
        x_max_check = x_max_check if x_max_check < self.discret_segment_size else self.discret_segment_size
        y_min_check = y_check - self.obs_range
        y_min_check = y_min_check if y_min_check > 0 else 0
        y_max_check = y_check + self.obs_range
        y_max_check = y_max_check if y_max_check < self.discret_segment_size else self.discret_segment_size

        nbr_cells = len(self.map[x_min_check:x_max_check, y_min_check:y_max_check])

        f -= np.sum(self.map[x_min_check:x_max_check, y_min_check:y_max_check])
        if update_idleness:
            # Change all values to 0
            self.map[x_min_check:x_max_check, y_min_check:y_max_check] = 0

        return f, nbr_cells

    def observe(self, x, y, update_idleness=False):
        """

        :param x: coord x
        :param y: coord y
        :param update_idleness: True if we just evaluate without updating the idleness
        :return: sum of idleness (negative form)
        """
        f = 0

        # Let reposition x and y [-map_size;map_size] into the map frame [0; map_size²]
        x = x + self.segment_size / 2
        y = y + self.segment_size / 2

        x_check, y_check = int(x * self.discretization), int(y * self.discretization)
        # Count the number of ones in the interval

        # Verify interval
        x_min_check = x_check - self.obs_range
        x_min_check = x_min_check if x_min_check > 0 else 0
        x_max_check = x_check + self.obs_range
        x_max_check = x_max_check if x_max_check < self.discret_segment_size else self.discret_segment_size
        y_min_check = y_check - self.obs_range
        y_min_check = y_min_check if y_min_check > 0 else 0
        y_max_check = y_check + self.obs_range
        y_max_check = y_max_check if y_max_check < self.discret_segment_size else self.discret_segment_size

        f -= np.sum(self.map[x_min_check:x_max_check,
                    y_min_check:y_max_check])
        if update_idleness:
            # Change all values to 0
            self.map[x_min_check:x_max_check, y_min_check:y_max_check] = 0

        return f

    def observe_2points(self, xA, yA, xB, yB):
        # Build the line equation
        f = 0
        cells_saw = 0
        x, y = get_points_from_segment(xA, yA, xB, yB)
        for i in range(0, len(x)):
            f_obs, nbr_cells = self.observe_and_check_cells(x[i], y[i], update_idleness=True)
            f += f_obs
            cells_saw += nbr_cells

        return f, cells_saw

    def reset_random(self):
        self.map = np.random.randint(100, size=(self.discret_segment_size, self.discret_segment_size))

    def new_time(self):
        self.map += 1

    def get_map(self):
        return self.map

    def set_map(self, new_map):
        self.map = new_map

    def update_from_map(self, other_map):
        self.map = np.where(self.map > other_map, other_map, self.map)

    def display_map(self):
        map_copy = copy.copy(self.get_map())
        map_copy[map_copy > 255] = 255

        map_copy = map_copy.astype(np.uint8)
        map_copy = np.rot90(map_copy, k=1, axes=(0, 1))

        img = np.ones([self.discret_segment_size, self.discret_segment_size, 3])
        img[:, :, 0] = map_copy/255
        img[:, :, 1] = map_copy/255
        img[:, :, 2] = map_copy/255

        cv2.imshow('image', img)
        k = cv2.waitKey(33)
        if k == 27:  # Esc key to stop
            exit()
        '''
        plt.imshow(np.rot90(map_copy, k=1, axes=(0, 1)), interpolation='nearest')
        plt.show(block=False)
        plt.pause(0.00001)
        '''
