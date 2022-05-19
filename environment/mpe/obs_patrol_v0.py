from pettingzoo.utils import wrappers
from ._mpe_utils.obs_patrol_env import SimpleEnv, make_env
from .scenarios.obs_patrol import Scenario
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv):
    def __init__(self, nbr_agent=2, nbr_target=2, obs_range=1, safety_range=2, dangerous_range=1,
                 map_size=10, com_range=2, obs_to_normalize=False, max_cycles=25, has_protection_force=False,
                 is_reward_individual=True, gradual_reward=True, is_reward_shared=False, max_target_speed=1,
                 max_agent_speed=1, share_target=False):
        scenario = Scenario()
        world = scenario.make_world(nbr_agent=nbr_agent, nbr_target=nbr_target, obs_range=obs_range,
                                    com_range=com_range, safety_range=safety_range, dangerous_range=dangerous_range,
                                    obs_to_normalize=obs_to_normalize, map_size=map_size,
                                    is_reward_individual=is_reward_individual, gradual_reward=gradual_reward,
                                    is_reward_shared=is_reward_shared, share_target=share_target, max_cycles=max_cycles)
        super().__init__(scenario, world, max_cycles, has_protection_force=has_protection_force,
                         max_target_speed=max_target_speed, max_agent_speed=max_agent_speed)
        self.metadata['name'] = "obs_patrol_v0"
        self.metadata['render_modes'] = "human"




env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
