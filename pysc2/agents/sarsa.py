from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import pandas as pd
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features    

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id

    
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

COLLECT_MINERAL_REWARD = 0.2

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectarmy'

smart_actions =[
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
	ACTION_MOVE_UNIT
]


class SARSAlearning_CollectMineralShards(base_agent.BaseAgent):
   def __init__(self):
        super(SARSAlearning_CollectMineralShards, self).__init__()
        self.sarsalearn = SarsaTable(actions=list(range(len(smart_actions))))
        self.previous_collected_minerals_score = 0
        self.previous_action = None
        self.previous_state = None
        self.rl_action = None
        
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def step(self, obs):
        super(SARSAlearning_CollectMineralShards, self).step(obs)
        
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()   #x,y coordinates of players
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        minerals_y, minerals_x = (unit_type == _PLAYER_NEUTRAL).nonzero()
        minerals_count = obs.observation['unit_counts'][1] #mineral counts is taken from observation

        current_state = [
            minerals_count,x            
            player_x,
            player_y
        ]

        if self.previous_action is not None:
            reward = 0
            if collected_minerals_score > self.previous_collected_minerals_score
                reward += COLLECT_MINERAL_REWARD
            self.sarsalearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state),self.rl_action)
        
        self.rl_action = self.sarsalearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]
        
        self.previous_collected_minerals_score = collected_minerals_score
        self.previous_state = current_state
        self.previous_action = rl_action

        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])
        
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
		
		elif smart_action == ACTION_MOVE_UNIT:
            if _MOVE_UNIT in obs.observation['available_actions']:
                return actions.FunctionCall(_MOVE_UNIT, [_NOT_QUEUED]) 
        
		return actions.FunctionCall(_NO_OP))
#This part is taken from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow and changed to the SARSA version
#Here is implementation of basic SARSA-learning algorithm
#I think own impelementation is not necessary in this point because this is already basic implementation of the SARSA pseudocode 
class SarsaTable:
    def __init__(self, actions, _epsilon=0.9, _max_steps=100, _alpha=0.85, _gamma = 0.95, _total_episodes = 10000):
        self.actions = actions  # a list
		self.epsilon = _epsilon
		self.max_steps = _max_steps
		self.alpha = _alpha
        self.gamma = _gamma
		self.total_episodes = _total_episodes 
        self.sarsa_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose best action according to sarsa table 
            state_action = self.sarsa_table.ix[observation, :]
            # some actions have the same value and they are reindexed
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action
	

    def learn(self, s, a, reward, s_, a_):
        self.check_state_exist(s_)					#Check the state is exist => S(t+1)
        self.check_state_exist(s)					#Check the state is exist => St
        sarsa_predict = self.sarsa_table.ix[s, a]	#Q(St,At) 
        sarsa_target = reward + self.gamma * self.sarsa_table.ix[s_,a_]	# Reward + Gamma * Q( S(t+1),A(t+1) )
        # updating sarsa table 
        self.sarsa_table.ix[s, a] += self.alpha * (sarsa_target - sarsa_predict) #Q(St,At) = Q(St,At) + LearningRate*(Reward(t+1) + Gamma * Q(S(t+1),A(t+1) - Q(St,At))

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to SARSA table
            self.sarsa_table = self.sarsa_table.append(pd.Series([0] * len(self.actions), index=self.sarsa_table.columns, name=state))


