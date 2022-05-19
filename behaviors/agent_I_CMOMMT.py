import math
import random

import cv2
import numpy as np

from behaviors.Tools.Map import Map
from behaviors.Tools.PF import PotentialFieldPatrol
from behaviors.Tools.functions import get_discrete_angle, get_discrete_speed
from behaviors.agent_A_CMOMMT import AgentACMOMMT


class AgentICMOMMT(AgentACMOMMT):
    """
    I-CMOMMT method
    """

    def __init__(self, map_size, obs_range=1, com_range=1, discretization=5, gamma=1):
        # Pose
        super().__init__(map_size, obs_range, com_range)

        self.obs_range = obs_range
        self.discretization = discretization
        self.gamma = gamma

        # Perform mask
        self.kernel_size = self.obs_range * self.discretization * 2
        self.kernel = np.ones(self.kernel_size) * 1 / self.kernel_size

        self.myMap = Map(map_size=map_size, obs_range=obs_range, discretization=discretization)

        self.PF = PotentialFieldPatrol(do1=self.do1, do2=self.do2, do3=self.do3, do4=self.do4,
                                       dr1=self.dr1, dr2=self.dr2, map=self.myMap)

        # Init
        random.seed()
        #self.x_des, self.y_des = self.get_random_point()
        self.index_x_des, self.index_y_des = 1, 1  # Init, no importance about first value
        self.update_angle()

    def desired_point(self):
        # Follow potential field attraction
        angle, norm = self.PF.get_resultante()
        if norm > 1:
            norm = 1
        self.angle = angle
        self.check_angle()
        self.speed = 1

    def add_target(self, x, y):
        self.PF.add_target(x, y)

    def add_agent(self, x, y):
        self.PF.add_agent(-x, -y)

    def patrol_force(self):
        c = cv2.filter2D(self.myMap.get_map(), -1, self.kernel)
        cmax_old = int(c[self.index_x_des, self.index_y_des])
        cmax = int(c.max())
        if cmax_old < cmax:  # Another better area to check seen
            cmax = c.max()
            index = np.where(c == cmax)
            if len(index[0]) > 1:
                r = random.randint(0, len(index[0]) - 1)
                self.index_x_des, self.index_y_des = index[0][r], index[1][r]
            elif len(index[0] == 1):
                self.index_x_des, self.index_y_des = index[0][0], index[1][0]
            else:
                print("PROBLEM ABOUT INDEX IN CMAX")

        self.x_des = (self.index_x_des / self.discretization) - self.map_size
        self.y_des = (self.index_y_des / self.discretization) - self.map_size #Recadrage autour de (0,0)
        magnitude = 1  # TODO : This is a first try, set a limit to contain f_p btw 0 and 1 ?

        self.PF.add_patrolling_force(self.x_des, self.y_des, magnitude)

    # COM'S
    def update_map_from_other(self, other_map):
        self.myMap.update_from_map(other_map)

    def get_map(self):
        return self.myMap.get_map()

    def get_action(self):
        self.myMap.observe(self.x, self.y, update_idleness=True)

        # Check lambda value
        lambda_ = math.tanh(np.max(self.myMap.get_map()) / self.gamma)
        self.PF.set_lambda(lambda_)

        # Compute the best area to go and set it to PF:
        if lambda_ > 0:
            self.patrol_force()

        self.desired_point()
        self.PF.reset()

        self.myMap.new_time()
        x_speed = self.speed * math.cos(self.angle)
        y_speed = self.speed * math.sin(self.angle)
        #self.myMap.display_map()
        return ([get_discrete_speed(x_speed), get_discrete_speed(y_speed)])

    def set_observation(self, observation):
        super().set_observation(observation)
        self.PF.set_pose(self.x, self.y)
