import copy
import math

import cv2
import numpy as np

from behaviors.Tools.Map import Map


class Analyze:

    def __init__(self, obs_range, discretization, obs_to_normalize, map_size, nbr_target):
        self.time = 0
        self.idleness_sum = 0
        self.nbr_cells = 0
        self.obs_range = obs_range
        self.obs_to_normalize = obs_to_normalize
        self.map_size = map_size
        self.discretization = discretization
        self.nbr_target = nbr_target

        # For the diversity function
        self.obs_i = np.zeros(self.nbr_target)
        self.target_seen = 0  # g function from matrix A in CMOMMT

        self.kernel_size = obs_range * discretization * 2

        self.max_idleness = 0
        self.max_idleness_blur = 0

        self.kernel = np.ones(self.kernel_size) * 1 / self.kernel_size

        self.list_pose_agent = []
        self.list_pose_target = []
        self.seen_agent = []

        self.map = Map(map_size=self.map_size, obs_range=self.obs_range, discretization=discretization)
        self.set_nbr_cells(self.map.get_map().size)

    def get_distance(self, poseA, poseB):
        xA, yA = poseA[0], poseA[1]
        xB, yB = poseB[0], poseB[1]
        return math.sqrt(math.pow(xA - xB, 2) + math.pow(yA - yB, 2))

    def update_states(self):
        i = 0
        for pose_target in self.list_pose_target:
            for pose_agent in self.list_pose_agent:
                if self.get_distance(pose_agent, pose_target) <= self.obs_range:
                    self.new_target_seen()
                    self.obs_i[i] = self.obs_i[i] + 1  # Target i is seen during this cycle
                    break
            i = i + 1

        self.seen_agent = []
        self.list_pose_agent = []
        self.list_pose_target = []
        self.new_time()

    def reset(self):
        self.seen_agent = []
        self.list_pose_agent = []
        self.list_pose_target = []
        self.time = 0
        self.target_seen = 0
        self.max_idleness = 0
        self.max_idleness_blur = 0
        self.idleness_sum = 0
        self.obs_i = np.zeros(self.nbr_target)
        self.map = Map(map_size=self.map_size, obs_range=self.obs_range, discretization=self.discretization)

    def new_target_seen(self):
        self.target_seen += 1

    def set_nbr_cells(self, a):
        self.nbr_cells = a

    def new_time(self):
        self.time += 1
        self.map.new_time()
        self.update_patrolling_metrics()

    def update_patrolling_metrics(self):
        actual_map = copy.copy(self.map.get_map())
        self.idleness_sum += np.sum(actual_map)
        max_idleness = np.max(actual_map)
        if max_idleness > self.max_idleness:
            self.max_idleness = max_idleness

        c = cv2.filter2D(actual_map, -1, self.kernel)
        cmax = np.max(c)
        if cmax > self.max_idleness_blur:
            self.max_idleness_blur = cmax

    def add_pose(self, observation, agent):
        # Check if the agent hasn't been already seen
        if agent in self.seen_agent:
            self.update_states()

        self.seen_agent.append(agent)
        x, y = observation['a_self_pose'][0]

        if "agent" in agent:
            if self.obs_to_normalize:
                x = x * self.map_size
                y = y * self.map_size
            self.list_pose_agent.append((x, y))
            self.map.observe(x, y, update_idleness=True)
        else:
            self.list_pose_target.append((x, y))

    # GETTER FOR METRICS
    def get_max_idleness(self):
        return self.max_idleness

    def get_o_av_m(self):
        o_av_m = self.idleness_sum / (self.nbr_cells * self.time)
        return o_av_m

    def get_max_idleness_blur(self):
        return self.max_idleness_blur

    def get_observability_A(self):
        return self.target_seen / self.time

    def get_diversity(self):
        S = np.sum(self.obs_i)
        if S <= 0:
            return 0

        p = self.obs_i / S
        H = 0
        for p_i in p:
            if p_i < 1e-10: #equal zero
                p_i = 1e-10
            H += p_i*math.log2(p_i)
        return -1*H