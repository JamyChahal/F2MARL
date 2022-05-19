import math
import random


class TargetBehavior:

    def __init__(self, map_size, id, max_speed):

        self.x = 0
        self.y = 0
        self.speed = max_speed

        self.des_x = 0
        self.des_y = 0
        self.map_size = map_size
        self.id = id
        self.new_random_des_pose()

    def get_id(self):
        return self.id

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
        #Try to be closer to the desired angle
        angle_discrete = self.get_discrete_angle(angle)
        return (self.speed, angle_discrete)

    def get_discrete_angle(self, angle):
        # [no_action, move_left, move_right, move_down, move_up]
        if angle <= math.pi and angle > 3/4*math.pi:
            return 1
        if angle <= 3/4*math.pi and angle > math.pi/4:
            return 4
        if angle <= math.pi/4 and angle > -math.pi/4:
            return 2
        if angle <= -math.pi/4 and angle > -5/4*math.pi:
            return 3
        if angle <= -5/4*math.pi and angle >= -math.pi:
            return 1
        else:
            return 0

    def set_observation(self, observation):
        for x in observation:
            if x[0] == 0:
                self.x, self.y = x[1], x[2]
                break