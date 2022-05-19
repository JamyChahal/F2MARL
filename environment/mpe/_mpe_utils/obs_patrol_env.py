import math

from gym import spaces
# from ray.rllib.utils.spaces.repeated import Repeated
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gym.utils import seeding
from pettingzoo.utils import wrappers


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        # env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


MAX_SEEN_TARGET = 8
MAX_SEEN_AGENT = 8


class SimpleEnv(AECEnv):
    def __init__(self, scenario, world, max_cycles, has_protection_force=False, local_ratio=None,
                 max_target_speed=1, max_agent_speed=1):
        super(SimpleEnv, self).__init__()

        self.seed()

        self.max_target_speed = max_target_speed
        self.max_agent_speed = max_agent_speed

        self.metadata = {'render.modes': ['human', 'rgb_array']}

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.local_ratio = local_ratio
        self.has_protection_force = has_protection_force

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {agent.name: idx for idx, agent in enumerate(self.world.agents)}

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        for agent in self.world.agents:
            '''#Discrete
            self.action_spaces[agent.name] = spaces.Tuple((
                spaces.Discrete(4),  # Speed
                spaces.Discrete(5))  # Angle
            )
            '''
            '''
            self.action_spaces[agent.name] = spaces.Tuple((
                spaces.Discrete(11),  # Speed
                spaces.Discrete(11))  # Angle
            )
            '''
            '''
            self.action_spaces[agent.name] = spaces.Tuple((
                spaces.Box(low=-np.float32(1), high=+np.float32(1), shape=(1,)),  # X, Y speed
                spaces.Box(low=-np.float32(1), high=+np.float32(1), shape=(1,)))  # Y speed
            )
            '''
            #self.action_spaces[agent.name] = spaces.Box(low=-np.array([1, 1]), high=+np.array([1, 1]), shape=(2,),
            #                                            dtype=np.float32)  # X, Y speed

            self.action_spaces[agent.name] = spaces.Tuple((
                spaces.Discrete(5),  # X
                spaces.Discrete(5)   # Y
            ))

            # self.action_spaces[agent.name] = spaces.Box(low=np.float32(0), high=np.float32(1), shape=(1,), dtype=np.float32)

            '''
            self.action_spaces[agent.name] = spaces.Tuple((
                spaces.Box(low=np.float32(0), high=np.float32(2*math.pi), shape=(1,), dtype=np.float32),
                spaces.Box(low=np.float32(0), high=np.float32(1), shape=(1,), dtype=np.float32)))
            '''
            pose_dim = 2

            DICT_SPACE = spaces.Dict({
                "a_self_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                          shape=(1, pose_dim), dtype=np.float32),
                "b_agent_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                           shape=(MAX_SEEN_AGENT, pose_dim), dtype=np.float32),
                "c_target_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                            shape=(MAX_SEEN_TARGET, pose_dim), dtype=np.float32),
                "d_all_pose": spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf),
                                         shape=(len(self.agents), pose_dim), dtype=np.float32)
            })  # a,b,c because it is an alphabetic order in rllib
            # self.observation_spaces[agent.name] = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(obs_dim,3), dtype=np.float32)
            self.observation_spaces[agent.name] = DICT_SPACE
        self.steps = 0

        self.current_actions = [None] * self.num_agents

        self.viewer = None

    # For ray2.0
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        # return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world).astype(np.float32)
        return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world)

    def reset(self):
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0. for name in self.agents}
        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self._reset_render()

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                '''
                mdim = self.world.dim_p * 2 + 1
                scenario_action.append(action % mdim)
                action //= mdim
                '''
                scenario_action.append(action)
            if not agent.silent:
                scenario_action.append(action)

            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = global_reward * (1 - self.local_ratio) + agent_reward * self.local_ratio
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # Set the discrete action space into true 2D motions
    def get_continuous_action(self, x_speed_choice, y_speed_choice):
        if x_speed_choice == 0:
            x_speed = 1
        elif x_speed_choice == 1:
            x_speed = 0.5
        elif x_speed_choice == 2:
            x_speed = 0
        elif x_speed_choice == 3:
            x_speed = -0.5
        elif x_speed_choice == 4:
            x_speed = -1
        else:
            x_speed = 0

        if y_speed_choice == 0:
            y_speed = 1
        elif y_speed_choice == 1:
            y_speed = 0.5
        elif y_speed_choice == 2:
            y_speed = 0
        elif y_speed_choice == 3:
            y_speed = -0.5
        elif y_speed_choice == 4:
            y_speed = -1
        else:
            y_speed = 0

        return x_speed, y_speed

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            # process discrete action
            x_speed_choice = action[0][0]
            y_speed_choice = action[0][1]

            x_speed, y_speed = self.get_continuous_action(x_speed_choice, y_speed_choice)

            if agent.adversary:
                x_speed = x_speed * self.max_target_speed
                y_speed = y_speed * self.max_target_speed
            else:
                x_speed = x_speed * self.max_agent_speed
                y_speed = y_speed * self.max_agent_speed

            agent.action.u[0] = x_speed
            agent.action.u[1] = y_speed

            """
            if not agent.adversary and self.has_protection_force:
                x_speed, y_speed = self.scenario.apply_protection_force(agent, self.world, x_speed, y_speed)
                # Override the command
                agent.action.u[0] = x_speed
                agent.action.u[1] = y_speed
            """

            ''' 
            speed = action[0][0]
            angle = action[0][1]
            agent.action.u[0] = speed * math.cos(angle)
            agent.action.u[1] = speed * math.sin(angle)
            '''
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            agent.action.c = np.zeros(self.world.dim_c)

            agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.dones[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

    def render(self, mode='human'):
        from . import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color[:3], alpha=0.5)
                else:
                    geom.set_color(*entity.color[:3])
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    tline = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(tline)
                    idx += 1

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # for agent in self.world.agents:
        idx = 0
        for idx, other in enumerate(self.world.agents):
            if other.silent:
                continue
            if np.all(other.state.c == 0):
                word = '_'
            else:
                word = alphabet[np.argmax(other.state.c)]

            message = (other.name + ' sends ' + word + '   ')

            self.viewer.text_lines[idx].set_text(message)
            idx += 1

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()
