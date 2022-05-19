import numpy as np
from gym import spaces

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()


class ObservationCentralizedCriticModel(TFModelV2):
    """Multi-agent model that implements a centralized value function.
    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).
    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(ObservationCentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        # self_obs_space = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
        #                                  shape=(17, 2), dtype=np.float32)
        self_obs_space = spaces.Box(low=-1, high=+1,
                                    shape=(17, 2), dtype=np.float32)

        self.action_model = FullyConnectedNetwork(
            self_obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_action")

        #all_obs_space = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
        #                           shape=(15, 2), dtype=np.float32)
        all_obs_space = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                   shape=(16, 2), dtype=np.float32)
        # 10 targets, 5 agents
        # Used to be 8 : 4 agents, 4 targets

        self.value_model = FullyConnectedNetwork(all_obs_space, action_space, 1,
                                                 model_config, name + "_vf")

    def forward(self, input_dict, state, seq_lens):
        self._value_out, _ = self.value_model({
            "obs": input_dict["obs"]["d_all_pose"]
        }, state, seq_lens)
        own_obs = tf.concat([input_dict["obs"]["a_self_pose"], input_dict["obs"]["b_agent_pose"],
                             input_dict["obs"]["c_target_pose"]], axis=1)

        return self.action_model({
            "obs": own_obs,
        }, state, seq_lens)

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
