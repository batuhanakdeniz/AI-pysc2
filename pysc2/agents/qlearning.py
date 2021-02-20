from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features    

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_MOVE_UNIT = actions.FUNCTIONS.moveunit.id

    
_PLAYER_SELF = features.PlayerRelative.SELF        #Players units like marine
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals for our minimap 
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index   #For understand which unit is mineral or marine

FUNCTIONS = actions.FUNCTIONS           #Functions in actions defined as FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS   #Raw Functions in actions defined as RAW_FUNCTIONS

COLLECT_MINERAL_REWARD = 0.2            #Reward value for Q-Learning

ACTION_DO_NOTHING = 'donothing'        
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_MOVE_UNIT = 'moveunit'

smart_actions =[                        #Actions are defined
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_MOVE_UNIT
]


class Qlearning_CollectMineralShards(base_agent.BaseAgent):
   def __init__(self):
        super(Qlearning_CollectMineralShards, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_collected_minerals_score = 0
        self.previous_action = None
        self.previous_state = None          #previous state and actions should be null for begining
        
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
    
    def step(self, obs):
        super(Qlearning_CollectMineralShards, self).step(obs)
        
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()   #x,y coordinates of players
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        unit_type = obs.observation['screen'][_UNIT_TYPE]               #Understanding which unit is this 
        minerals_y, minerals_x = (unit_type == _PLAYER_NEUTRAL).nonzero()   #If neutral, take minerals coordinates
        minerals_count = obs.observation['unit_counts'][1] #mineral counts is taken from observation

        current_state = [       #Defining the state's informations
            minerals_count,
            player_x,
            player_y
        ]

        if self.previous_action is not None:    #If previous action is not None
            reward = 0                          #Reward equals to 0
            if collected_minerals_score > self.previous_collected_minerals_score    #p
                reward += COLLECT_MINERAL_REWARD                                    #Sum reward and reward constant
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))   #Q-Learn with previous state, previous action, 
                                                                                                            #reward and current state
        rl_action = self.qlearn.choose_action(str(current_state))
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

#This part is taken from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
#Here is implementation of basic Q-learning algorithm
#I think own impelementation is not necessary in this point because this is already basic implementation of the Q learning pseudocode 
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):#Initilizing the required variables 
        self.actions = actions  # action list
        self.lr = learning_rate 
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):   #Choosing the action according to states
        self.check_state_exist(observation)   
        if np.random.uniform() < self.epsilon:
            # choose best action in the q table 
            state_action = self.q_table.ix[observation, :]
            # some actions have the same value and they are reindexed
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            #action equals to best action 
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):       #Learn function 
        self.check_state_exist(s_)      #Check the state is exist => S(t+1)
        self.check_state_exist(s)       #Check the state is exist => St
        q_predict = self.q_table.ix[s, a]   #Q(St,At)
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()    #(Rt + Gamma * max(Q(S(t+1),For all A))
        # updating Q(St,At)
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)   #Q(St,At) = Q(St,At) + LearningRate*(Reward(t+1) + Gamma * max(Q(S(t+1)))

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))