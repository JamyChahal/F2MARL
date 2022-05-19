import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AgentBehavior:

    def __init__(self, map_size, id, max_speed):

        self.x = 0
        self.y = 0
        self.max_speed = max_speed

        self.des_x = 0
        self.des_y = 0
        self.map_size = map_size
        self.id = id
        self.reward = 0

        self.new_random_des_pose()

    def get_id(self):
        return self.id

    def set_reward(self, r):
        #print("Agent "+ str(self.get_id())+" has r="+str(r))
        self.reward += r

    def compute_reward(self):
        # TODO : Set the reward to the LSTM
        self.reward = 0

    def random_pose(self):
        x = random.randint(-self.map_size, self.map_size)
        y = random.randint(-self.map_size, self.map_size)
        return x, y

    def new_random_des_pose(self):
        self.des_x, self.des_y = self.random_pose()

    def distance_to_des_pose(self):
        return math.sqrt(math.pow(self.des_x - self.x, 2)+math.pow(self.des_y - self.y, 2))

    def get_action(self):
        d = self.distance_to_des_pose()
        if d < 0.5:
            self.new_random_des_pose()
        angle = math.atan2(self.des_y - self.y, self.des_x - self.x)
        # Try to be closer to the desired angle
        angle_discrete = self.get_discrete_angle(angle)
        return (self.max_speed, angle_discrete)

    def get_discrete_angle(self, angle):
        # [no_action, move_left, move_right, move_down, move_up]
        if math.pi >= angle > 3 / 4 * math.pi:
            return 1
        if 3 / 4 * math.pi >= angle > math.pi / 4:
            return 4
        if math.pi / 4 >= angle > -math.pi / 4:
            return 2
        if -math.pi / 4 >= angle > -5 / 4 * math.pi:
            return 3
        if -5 / 4 * math.pi >= angle >= -math.pi:
            return 1
        else:
            return 0

    def set_observation(self, observation):
        for x in observation:
            if x[0] == 0:
                self.x, self.y = x[1], x[2]
                break #TODO : Add friends and targets