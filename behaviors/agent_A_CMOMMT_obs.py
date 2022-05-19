import math
import random

import numpy as np

from behaviors.Tools.PF import PotentialField
from behaviors.Tools.functions import get_discrete_angle, get_discrete_speed


class AgentACMOMMTobs:
    """
    A-CMOMMT method without motion if no targets
    """

    def __init__(self, map_size, obs_range=1, com_range=1):

        random.seed()
        # Pose
        self.x, self.y = 0, 0
        self.x_des, self.y_des = 0, 0
        self.angle = 0

        self.map_size = map_size
        ''' Previously
        self.do1 = obs_range / 4
        self.do2 = obs_range / 3
        self.do3 = obs_range / 2
        '''
        self.do1 = 1
        self.do2 = 2
        self.do3 = 4
        self.do4 = 5
        self.dr1 = 1
        self.dr2 = 2

        self.PF = PotentialField(do1=self.do1, do2=self.do2, do3=self.do3, do4=self.do4,
                                 dr1=self.dr1, dr2=self.dr2)

        # Init
        self.x, self.y = 0, 0
        self.prev_pose = []
        self.update_angle()

        self.speed = 1
        self.timer_period = 1  # sec
        self.target_seen = False

    def get_pose(self):
        return self.x, self.y

    def update_angle(self):
        self.angle = math.atan2(self.y_des - self.y, self.x_des - self.x)

    def get_distance_to_goal(self):
        return math.sqrt(math.pow(self.x - self.x_des, 2) + math.pow(self.y - self.y_des, 2))

    def check_angle(self):
        # Check if angle is between -pi and pi
        while self.angle > math.pi:
            self.angle -= 2 * math.pi
        while self.angle < -math.pi:
            self.angle += 2 * math.pi

    def random_start(self):
        self.x, self.y = self.get_random_point()

    def get_random_point(self):
        x = random.randint(-self.map_size, self.map_size)
        y = random.randint(-self.map_size, self.map_size)
        return x, y

    '''
    def desired_point(self):
        if self.PF.is_empty():
            # Random walk
            self.speed = 1  # Has to be changed if strategy changed
            if not self.PF.is_still_empty() or self.get_distance_to_goal() < 0.5:
                # Add a new random target point
                self.x_des, self.y_des = self.get_random_point()
                self.check_angle()
                self.update_angle()

        else:  # Follow potential field attraction
            angle, norm = self.PF.get_resultante()
            self.angle = angle
            self.check_angle()
            self.speed = norm * self.max_velocity
    '''
    def update_pose_history(self):
        self.prev_pose.append((self.x, self.y))
        if len(self.prev_pose) > 5:
            self.prev_pose.pop(0)

    def is_agent_stuck(self):
        if len(self.prev_pose) > 4:
            if self.prev_pose.count(self.prev_pose[0]) == len(self.prev_pose):
                return True
            else:
                return False
        else:
            return False

    def desired_point(self):
        if self.PF.is_empty() or not self.target_seen:
            # Add random direction
            angle, norm = 0, 0
        else:
            # Follow potential field attraction
            angle, norm = self.PF.get_resultante()
        self.angle = angle
        self.check_angle()
        self.speed = norm

    def add_target(self, x, y):
        self.PF.add_target(x, y)
        self.target_seen = True

    def add_agent(self, x, y):
        self.PF.add_agent(x, y)

    def add_destination(self, x, y):
        self.PF.add_destination(x, y)

    def get_action(self):
        self.update_angle()
        self.check_angle()

        self.desired_point()
        self.PF.reset()

        self.target_seen = False
        x_speed = self.speed * math.cos(self.angle)
        y_speed = self.speed * math.sin(self.angle)

        return ([get_discrete_speed(x_speed), get_discrete_speed(y_speed)])

    def set_observation(self, observation):
        self.x, self.y = observation['a_self_pose'][0]
        for agent in observation['b_agent_pose']:
            if abs(agent[0]) > 0 or abs(agent[1]) > 0:
                self.add_agent(agent[0], agent[1])
        for target in observation['c_target_pose']:
            if abs(target[0]) > 0 or abs(target[1]) > 0:
                self.add_target(target[0], target[1])

