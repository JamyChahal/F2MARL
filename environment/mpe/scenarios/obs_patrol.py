import copy
import math
from collections import OrderedDict

import numpy as np

from behaviors.Tools.functions import get_discrete_angle, get_discrete_speed
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self, nbr_agent=2, nbr_target=0, obs_range=1, com_range=2, map_size=10, safety_range=2,
                   dangerous_range=1, obs_to_normalize=False, is_reward_individual=False, is_reward_shared=False,
                   gradual_reward=True, share_target=False, max_cycles=25):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = nbr_agent + nbr_target
        world.num_agents = num_agents
        self.num_adversaries = nbr_target
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < self.num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < self.num_adversaries else i - self.num_adversaries
            agent.name = '{}_{}'.format(base_name, base_index)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # add other parameters
        self.max_cycles = max_cycles
        self.obs_range = obs_range
        self.com_range = com_range
        self.map_size = map_size
        self.obs_to_normalize = obs_to_normalize  # If the obs has to normalize between [-1 1]
        self.is_reward_individual = is_reward_individual
        self.gradual_reward = gradual_reward
        self.is_reward_shared = is_reward_shared
        self.share_target = share_target

        # For protection force variables
        self.sr = safety_range
        self.dr = dangerous_range
        if abs(self.sr - self.dr) < 0.0001:
            self.sr = self.dr + 1 # Avoid zero division

        self.fr1a = 1 / float(self.sr - self.dr)
        self.fr1b = self.sr#-1 - self.fr1a * self.dr1
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i in range(0, world.num_agents):
            if i < self.num_adversaries:
                world.agents[i].color = np.array([0.85, 0.35, 0.35])
            else:
                world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-self.map_size, +self.map_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return 0  # TODO
        else:
            dists = []
            # TODO
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on observation
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def get_distance(self, agent1, agent2):
        return math.sqrt(math.pow(agent1.state.p_pos[0] - agent2.state.p_pos[0], 2) +
                         math.pow(agent1.state.p_pos[1] - agent2.state.p_pos[1], 2))

    def agent_reward(self, agent, world):
        '''
        Return the reward for the agent
        +1 for each target seen by at least one agent
        -100 if too close to another agent : Not anymore
        -100 if outside the world : Not anymore
        '''

        reward = 0
        adversary_agents = self.adversaries(world)
        good_agents = self.good_agents(world)

        if self.is_reward_individual:
            for target in adversary_agents:
                if self.get_distance(target, agent) <= self.obs_range:
                    reward += 1

        if not self.is_reward_shared and not self.is_reward_individual:
            for target in adversary_agents:
                # If the target is seen by me
                if self.get_distance(target, agent) <= self.obs_range:
                    nbr_agent_seeing_target = 1
                    for ga in good_agents:
                        if agent is ga:
                            continue
                        if self.get_distance(target, ga) <= self.obs_range:
                            nbr_agent_seeing_target += 1
                    reward += 1 / nbr_agent_seeing_target

        if self.is_reward_shared and not self.is_reward_individual:
            for target in adversary_agents:
                for ga in good_agents:
                    if self.get_distance(target, ga) <= self.obs_range:
                        reward += 1/(self.max_cycles*len(adversary_agents))  # Definition of the normalized A metric (sum(obs)/sum(times))
                        break  # At least one agent see the target, now move on the next target

        # Collide negative reward
        for ga in good_agents:
            if agent is ga:
                continue
            dist_collide = self.get_distance(ga, agent)
            if self.sr <= dist_collide < self.dr and self.gradual_reward:
                reward -= 1/(self.sr - self.dr) * dist_collide - self.sr/(self.sr-self.dr)  # ax+b
            if dist_collide <= self.dr:
                reward -= 1 # TODO : Divide it by the number of max cycle ?

        # Outside the map
        '''
        if self.gradual_reward:
            out = self.how_far_agent_outside_map(agent)
            if out > 0:
                reward -= 20*out
            if self.is_agent_outside_map(agent, 10):
                reward -= 500  # Outside of the map, don't go there
        else:
            if self.is_agent_outside_map(agent, 1):
                reward -= 500
        '''
        return reward

    def is_agent_outside_map(self, agent, limit):
        if agent.state.p_pos[0] > self.map_size + limit or \
                agent.state.p_pos[0] < -self.map_size - limit or \
                agent.state.p_pos[1] > self.map_size + limit or \
                agent.state.p_pos[1] < -self.map_size - limit:
            print("Agent outside !")
            return True
        else:
            return False

    def how_far_agent_outside_map(self, agent):
        x_max = agent.state.p_pos[0] - self.map_size
        x_min = -self.map_size - agent.state.p_pos[0]
        y_max = agent.state.p_pos[1] - self.map_size
        y_min = -self.map_size - agent.state.p_pos[1]
        out = max(x_max, 0) + max(x_min, 0) + max(y_max, 0) + max(y_min, 0)
        return out

    def apply_protection_force(self, agent, world, x_speed, y_speed):
        def get_robot_magnitude(distance):
            m = 0
            if distance <= self.dr:
                m = -1
            if self.dr < distance <= self.sr:
                m = self.fr1a * distance + self.fr1b
            if distance > self.sr:
                m = 0
            return m

        x_coord, y_coord = x_speed, y_speed

        # Repulsion from the outside world
        x_force, y_force = 0, 0
        limit = 1
        if agent.state.p_pos[0] - self.map_size > limit:
            x_force = -1
        if (-self.map_size - agent.state.p_pos[0]) > limit:
            x_force = 1
        if agent.state.p_pos[1] - self.map_size > limit:
            y_force = -1
        if -self.map_size - agent.state.p_pos[1] > limit:
            y_force = 1

        # Repulsion from the other agents
        good_agents = self.good_agents(world)
        for ga in good_agents:
            if agent is ga:
                continue
            dist_collide = self.get_distance(ga, agent)
            if dist_collide <= self.sr:
                magnitude = get_robot_magnitude(dist_collide)
                vector = ga.state.p_pos - agent.state.p_pos
                x_force -= vector[0] * magnitude
                y_force -= vector[1] * magnitude
        # Final calculation
        x_final_force = x_coord + x_force
        y_final_force = y_coord + y_force

        return x_final_force, y_final_force

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        return 0  # For now, targets has no rewards

    def observation(self, agent, world):
        def sorting(pose_list):
            pose_list.sort(key=lambda x: (x[0] ** 2 + x[1] ** 2))
            return pose_list[0:8]

        def distance(relative_pose):
            return math.sqrt(relative_pose[0] ** 2 + relative_pose[1] ** 2)

        def remove_redundant_pose(seq):
            return list(OrderedDict((tuple(x), x) for x in seq).values())

        self_pos = [copy.copy(agent.state.p_pos)]
        agent_pose = []
        target_pose = []
        for other in world.agents:
            if other is agent:
                continue
            relative_pose = other.state.p_pos - agent.state.p_pos
            dist = math.sqrt(relative_pose[0] ** 2 + relative_pose[1] ** 2)
            if other.adversary:
                if dist <= self.obs_range:
                    target_pose.append(relative_pose)
            elif dist <= self.com_range:
                agent_pose.append(-1*relative_pose)  # TODO : Experimental

        # Communicate with the surrounding agent about targets (can be incorporated in the previous for loop)
        if np.shape(target_pose)[0] < 8 and self.share_target:
            for other in world.agents:
                if not other.adversary:
                    relative_pose = other.state.p_pos - agent.state.p_pos
                    dist = math.sqrt(relative_pose[0] ** 2 + relative_pose[1] ** 2)
                    if dist <= self.com_range:  # Within our communication range
                        agent_com_pose = other.state.p_pos
                        # Check its surrounding targets
                        for other_com in world.agents:
                            if other_com.adversary:
                                relative_com_pose = other_com.state.p_pos - agent_com_pose
                                if distance(relative_com_pose) <= self.com_range:
                                    target_pose.append(other_com.state.p_pos - agent.state.p_pos)

        # Limit it here for 4 agents and 4 targets, only the closest
        if np.shape(agent_pose)[0] > 8:
            agent_pose = sorting(agent_pose)
        if np.shape(target_pose)[0] > 8:
            target_pose = sorting(target_pose)

        # Remove redundant
        if self.share_target:
            target_pose = remove_redundant_pose(target_pose)

        # Completing to have specifically 4 agents and 4 targets info (with full 0)
        while np.shape(agent_pose)[0] < 8:
            agent_pose.append(np.array((0, 0)))
        while np.shape(target_pose)[0] < 8:
            target_pose.append(np.array((0, 0)))

        # Normalization of the self_pose (with map_size), agent_pose (with com_range) and target_pose (with obs_range)
        if not agent.adversary and self.obs_to_normalize:
            self_pos[0][0] = self_pos[0][0] / self.map_size
            self_pos[0][1] = self_pos[0][1] / self.map_size
            for i in range(0, len(agent_pose)):
                agent_pose[i][0] = agent_pose[i][0] / self.com_range
                agent_pose[i][1] = agent_pose[i][1] / self.com_range
            for i in range(0, len(target_pose)):
                target_pose[i][0] = target_pose[i][0] / self.obs_range
                target_pose[i][1] = target_pose[i][1] / self.obs_range

        all_pose = []  # world.agents is already sort (adversary then agent)
        for other in world.agents:
            all_pose.append(other.state.p_pos)

        obs = {
            "a_self_pose": self_pos,
            "b_agent_pose": agent_pose,
            "c_target_pose": target_pose,
            "d_all_pose": all_pose  # Only for centralized critic
        } #Sort by alphabetic, because rllib open the Dict in this way

        return obs
