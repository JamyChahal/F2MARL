import math
import random
import time

from ray.rllib.utils.framework import try_import_tf
import numpy as np

from behaviors.Tools.functions import get_discrete_speed

tf1, tf, tfv = try_import_tf()


class AgentNNRP:

    def __init__(self, sess, model, map_size):
        self.session = sess
        self.model = model

        # For model prediction
        self.is_training = np.array(0, dtype=np.int64)  # Set false
        self.prev_action = (1, 1)  # Init
        self.timestep = 0
        self.prev_reward = 0

        # self.session.run(tf1.global_variables_initializer())
        tf1.global_variables_initializer().run(session=self.session)

        # Try to do as : https://stackoverflow.com/questions/57672444/calling-tf-session-run-gets-slower
        # https://towardsdatascience.com/understanding-fundamentals-of-tensorflow-program-and-why-it-is-necessary-94cf5b60e255
        self.PH_is_training = tf1.placeholder(tf1.bool, name="is_training")
        self.PH_timestep = tf1.placeholder(tf1.int64, name="timestep")
        #self.PH_prev_reward = tf1.placeholder(tf1.float32, name="prev_reward")
        self.PH_observation = tf1.placeholder(tf1.float32, name="observation")
        #self.PH_prev_action = tf1.placeholder(tf1.float32, name="prev_action")

        self.prediction = self.model(
            is_training=self.PH_is_training,
            #prev_action=self.PH_prev_action,
            #prev_reward=self.PH_prev_reward,
            timestep=self.PH_timestep,
            observations=self.PH_observation
        )

        random.seed()
        self.map_size = map_size
        self.state = 0  # 0 : patroller, 1 : observer
        self.x, self.y = 0, 0
        self.x_des, self.y_des = self.get_random_point()

    def get_random_point(self):
        x = random.randint(-self.map_size, self.map_size) / self.map_size
        y = random.randint(-self.map_size, self.map_size) / self.map_size
        return x, y

    def distance_to_des_pose(self):
        return math.sqrt(math.pow(self.x_des - self.x, 2) + math.pow(self.y_des - self.y, 2))

    def do_action(self, reward, observation):
        self.x, self.y = observation['a_self_pose'][0]

        has_target = False
        for target in observation['c_target_pose']:
            if abs(target[0]) > 0 or abs(target[1]) > 0:
                has_target = True
                break

        if has_target:
            self.state = 1
            obs = self.vectorization_obs(observation, True)
            action = self.do_prediction(reward, obs)
        else:
            if self.distance_to_des_pose() < 0.5:
                self.x_des, self.y_des = self.get_random_point()
            self.state = 0
            angle = math.atan2(self.y_des - self.y, self.x_des - self.x)

            x_speed = math.cos(angle)
            y_speed = math.sin(angle)
            # Change into discrete action
            x_speed = get_discrete_speed(x_speed)
            y_speed = get_discrete_speed(y_speed)

            action = ([x_speed, y_speed])

        return action


    def vectorization_obs(self, obs, centralized_learning=True):
        observation = [obs['a_self_pose'][0][0], obs['a_self_pose'][0][1]]
        for x in obs['b_agent_pose']:
            observation.append(x[0])
            observation.append(x[1])
        for x in obs['c_target_pose']:
            observation.append(x[0])
            observation.append(x[1])

        if centralized_learning:  # Add fake pose, because cannot see it
            for x in range(0, 32):  # Trained by 4 agents and 8 targets, so 12x2 pose info
                observation.append(0)
        return np.array(observation, np.float32)

    def do_prediction(self, reward, observation):
        #print(self.model.structured_outputs)
        #print(self.model.inputs)

        result_output = self.session.run(self.prediction, feed_dict={
            self.PH_is_training: self.is_training,
            #self.PH_prev_action: np.array([self.prev_action], dtype=np.float32),
            #self.PH_prev_reward: [self.prev_reward],
            self.PH_timestep: self.timestep,
            self.PH_observation: [observation]
        })

        #action = tuple(((result_output['actions_0'][0][0]), (result_output['actions_0'][0][1])))
        action = tuple(((result_output['actions_0']), (result_output['actions_1'])))
        self.prev_action = action
        self.timestep += 1
        self.prev_reward = reward

        return action
