import json
import math
import random

import numpy as np
from gym.spaces import Box

from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights

from behaviors.randomBehavior import RandomBehavior


def make_randomBehavior(map_size):

    class RandomWalkingPolicy(Policy):
        """Hand-coded policy that returns random walking for training."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.behavior = RandomBehavior(map_size=map_size, is_in_training=True)

        @override(Policy)
        def compute_actions(self,
                            obs_batch,
                            state_batches=None,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            **kwargs):
            self.behavior.set_observation(obs_batch[0])
            # Action to perform here
            unbatched = [self.behavior.get_action()]

            actions = tuple(
                np.array([unbatched[j][i] for j in range(len(unbatched))]) for i in range(len(unbatched[0]))
            )

            #actions = unbatched  # Because not a tuple
            return actions, [], {}

        @override(Policy)
        def learn_on_batch(self, samples):
            """No learning."""
            return {}

        @override(Policy)
        def compute_log_likelihoods(self,
                                    actions,
                                    obs_batch,
                                    state_batches=None,
                                    prev_action_batch=None,
                                    prev_reward_batch=None):
            return np.array([random.random()] * len(obs_batch))

        @override(Policy)
        def get_weights(self) -> ModelWeights:
            """No weights to save."""
            return {}

        @override(Policy)
        def set_weights(self, weights: ModelWeights) -> None:
            """No weights to set."""
            pass

    return RandomWalkingPolicy
