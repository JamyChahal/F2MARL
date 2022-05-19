import math
import random

from behaviors.Tools.functions import get_discrete_angle


class OutsideBehavior:
    """
    Its purpose is to go outside the map. Used to check if env is resetting (done = True)
    """

    def __init__(self, map_size, id, max_speed, is_in_training=False):
        self.x = 0
        self.y = 0
        self.speed = max_speed
        self.des_x = 0
        self.des_y = 0
        self.map_size = map_size
        self.id = id
        self.is_in_training = is_in_training  # Is called by a training policy -> random_walking_policy.py

        random.seed()
        self.outside_pose()

    def get_id(self):
        return self.id

    def random_pose(self):
        x = random.randint(-self.map_size, self.map_size)
        y = random.randint(-self.map_size, self.map_size)
        return x, y

    def outside_pose(self):
        x = -self.map_size - 100
        y = -self.map_size - 100
        return x, y

    def outside_des_pose(self):
        self.des_x, self.des_y = self.outside_pose()

    def new_random_des_pose(self):
        self.des_x, self.des_y = self.random_pose()

    def distance_to_des_pose(self):
        return math.sqrt(math.pow(self.des_x - self.x, 2) + math.pow(self.des_y - self.y, 2))

    def get_action(self):
        d = self.distance_to_des_pose()
        if d < 0.5:
            self.outside_des_pose()
        angle = math.atan2(self.des_y - self.y, self.des_x - self.x)
        # Try to be closer to the desired angle
        # angle_discrete = get_discrete_angle(angle)

        x_speed = self.speed * math.cos(angle)
        y_speed = self.speed * math.sin(angle)

        return ([x_speed, y_speed])

    def set_observation(self, observation):
        if not self.is_in_training:
            self.x, self.y = observation['a_self_pose'][0]  # Considered as a dict
        else:
            self.x, self.y = observation[0], observation[1]  # Has been vectorized
