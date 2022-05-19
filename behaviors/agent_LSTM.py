import os
import time

from ray.rllib.utils.framework import try_import_tf
import numpy as np

tf1, tf, tfv = try_import_tf()


class AgentLSTM:

    def __init__(self, sess, model):
        self.session = sess
        self.model = model

        self.lstm_cell_size = 16  # 256

        tf1.global_variables_initializer().run(session=sess)

        # For model prediction
        self.is_training = np.array(0, dtype=np.int64)  # Set false
        self.prev_action = (1, 1)  # Init
        self.prev_reward = 0
        self.timestep = 0
        # self.init_state = np.asarray([np.zeros([self.lstm_cell_size], np.float32) for _ in range(1)])
        self.init_state = np.zeros(shape=[1, self.lstm_cell_size], dtype=np.float32)

        self.PH_is_training = tf1.placeholder(tf1.bool, name="is_training")
        self.PH_timestep = tf1.placeholder(tf1.int64, name="timestep")
        self.PH_prev_reward = tf1.placeholder(tf1.float32, name="prev_reward")
        self.PH_observation = tf1.placeholder(tf1.float32, name="observation")
        self.PH_prev_action = tf1.placeholder(tf1.float32, name="prev_action")
        self.PH_seq_lens = tf1.placeholder(tf1.int32, name="seq_lens")
        self.PH_placeholder_0 = tf1.placeholder(tf1.float32)
        self.PH_placeholder_1 = tf1.placeholder(tf1.float32)

        placeholders = {
            "is_training": self.PH_is_training,
            "observations": self.PH_observation,
            "seq_lens": self.PH_seq_lens,  # [20] wasn't working
            "prev_action": self.PH_prev_action,
            "prev_reward": self.PH_prev_reward,
            "timestep": self.PH_timestep,
            "policy_agent/Placeholder:0": self.PH_placeholder_0,
            "policy_agent/Placeholder_1:0": self.PH_placeholder_1
        }

        self.prediction = self.model(**placeholders)

    def do_prediction(self, reward, observation):
        # print(self.model.structured_outputs)
        # print(self.model.inputs)

        # Regarding the problem with Placeholder
        # TypeError: pruned(policy_agent/Placeholder:0, policy_agent/Placeholder_1:0, prev_action, is_training,
        # observations, prev_reward, seq_lens, timestep) missing required arguments: policy_agent/Placeholder:0,
        # policy_agent/Placeholder_1:0

        # Trying to have a look at :
        # https://discuss.ray.io/t/rllib-restoring-a-gtrxlnet-or-use-attention-true-fails/1699/2

        # self.session.run(tf1.global_variables_initializer())
        result_output = self.session.run(self.prediction, feed_dict={
            self.PH_is_training: self.is_training,
            self.PH_prev_action: np.array([self.prev_action], dtype=np.int64),
            self.PH_prev_reward: [self.prev_reward],
            self.PH_timestep: self.timestep,
            self.PH_observation: [observation],
            self.PH_seq_lens: [1],
            self.PH_placeholder_0: self.init_state,
            self.PH_placeholder_1: self.init_state
        })

        action = tuple(((result_output['actions_0'][0][0]), (result_output['actions_0'][0][1])))
        self.prev_action = action
        self.timestep += 1
        self.prev_reward = reward

        return action
