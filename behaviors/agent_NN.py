import time

from ray.rllib.utils.framework import try_import_tf
import numpy as np

tf1, tf, tfv = try_import_tf()


class AgentNN:

    def __init__(self, sess, model):
        self.session = sess
        self.model = model

        # For model prediction
        self.is_training = np.array(0, dtype=np.int64)  # Set false
        self.prev_action = (1, 1)  # Init
        self.timestep = 0
        self.prev_reward = 0

        #self.session.run(tf1.global_variables_initializer())
        tf1.global_variables_initializer().run(session=self.session)

        #Try to do as : https://stackoverflow.com/questions/57672444/calling-tf-session-run-gets-slower
        #https://towardsdatascience.com/understanding-fundamentals-of-tensorflow-program-and-why-it-is-necessary-94cf5b60e255
        self.PH_is_training = tf1.placeholder(tf1.bool, name="is_training")
        self.PH_timestep = tf1.placeholder(tf1.int64, name="timestep")
        self.PH_prev_reward = tf1.placeholder(tf1.float32, name="prev_reward")
        self.PH_observation = tf1.placeholder(tf1.float32, name="observation")
        self.PH_prev_action = tf1.placeholder(tf1.float32, name="prev_action")

        self.prediction = self.model(
            is_training=self.PH_is_training,
            prev_action=self.PH_prev_action,
            prev_reward=self.PH_prev_reward,
            timestep=self.PH_timestep,
            observations=self.PH_observation
        )

    def do_prediction(self, reward, observation):
        #print(self.model.structured_outputs)
        #print(self.model.inputs)

        result_output = self.session.run(self.prediction, feed_dict={
            self.PH_is_training: self.is_training,
            self.PH_prev_action: np.array([self.prev_action], dtype=np.float32),
            self.PH_prev_reward: [self.prev_reward],
            self.PH_timestep: self.timestep,
            self.PH_observation: [observation]
        })

        action = tuple(((result_output['actions_0'][0][0]), (result_output['actions_0'][0][1])))
        self.prev_action = action
        self.timestep += 1
        self.prev_reward = reward

        return action
