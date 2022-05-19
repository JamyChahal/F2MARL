import math
import random

from behaviors.Tools.PF import PotentialField, PotentialFieldTarget
from behaviors.Tools.functions import get_discrete_angle, get_discrete_speed


class TargetReactive:
    """
    Target avoiding seeing agents
    """

    def __init__(self, map_size, det_range=1, is_in_training=False):
        random.seed()
        # Pose
        self.x, self.y = 0, 0
        self.x_des, self.y_des = 0, 0
        self.angle = 0

        self.map_size = map_size

        self.PF = PotentialFieldTarget(det_range)

        # Init
        self.x, self.y = 0, 0
        self.prev_pose = []
        self.x_des, self.y_des = self.get_random_point()

        self.speed = 1
        self.timer_period = 1  # sec
        self.is_in_training = is_in_training
        self.agent_seen = False
        self.exit_step = 0

    def is_close_to_walls(self):
        epsilon = 2
        return self.is_close_to_walls_function(epsilon)

    def is_on_the_walls(self):
        epsilon = 1
        return self.is_close_to_walls_function(epsilon)

    def is_close_to_walls_function(self, epsilon):
        if self.x > self.map_size-epsilon or \
                self.y > self.map_size-epsilon or \
                self.x < -self.map_size+epsilon or \
                self.y < -self.map_size+epsilon:
            return True
        else:
            return False

    def get_pose(self):
        return self.x, self.y

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

    def reset_exit_step(self):
        self.exit_step = 60

    def update_exit_step(self):
        if self.exit_step > 0:
            self.exit_step = self.exit_step - 1

    def is_exiting(self):
        return True if self.exit_step > 0 else False

    def desired_point(self):
        self.update_pose_history()
        if self.get_distance_to_goal() < 0.5 or self.is_agent_stuck() or self.is_close_to_walls():
            self.x_des, self.y_des = self.get_random_point()
        self.add_destination(self.x_des - self.x, self.y_des - self.y)

        # Follow potential field attraction
        angle, norm = self.PF.get_resultante()
        self.angle = angle
        self.check_angle()
        if norm > 0:
            self.speed = 1
        else:
            self.speed = 0

        self.update_exit_step()
        if self.is_close_to_walls():
            self.reset_exit_step()

    def see_agent(self, x, y):
        # Ignore agents if too close to map boundaries
        if not self.is_exiting():
            self.PF.see_agent(-x, -y)

    def add_destination(self, x, y):
        self.PF.add_destination(x, y)

    def get_action(self):
        self.desired_point()
        self.PF.reset()

        #angle = get_discrete_angle(self.angle)
        #speed = get_discrete_speed(self.speed)

        if self.speed > 1:  # between 0 and 1
            self.speed = 1

        x_speed = math.cos(self.angle)
        y_speed = math.sin(self.angle)
        # Change into discrete action
        x_speed = get_discrete_speed(x_speed)
        y_speed = get_discrete_speed(y_speed)

        return ([x_speed, y_speed])

    def set_observation(self, observation):
        if not self.is_in_training:
            self.x, self.y = observation['a_self_pose'][0]  # Considered as a dict
            for agent in observation['b_agent_pose']:
                if abs(agent[0]) > 0 or abs(agent[1]) > 0:
                    self.see_agent(agent[0], agent[1])
        else:
            self.x, self.y = observation[0], observation[1]  # Has been vectorized
            for i in range(2, 10, 2):
                if abs(observation[i]) > 0 and abs(observation[i+1]) > 0:
                    self.see_agent(observation[i], observation[i+1])


