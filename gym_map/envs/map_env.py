import gym
from gym import spaces
from gym.utils import seeding
from gym_map.envs.map_view import Map
from gym_map.envs.map_constants import (DEFAULT_MAP_DATA, MAX_WIDTH,
    MAX_HEIGHT, MAP_BOUNDS, MAX_CHECKPOINTS, MAX_TELEPORTS, TILE_DICT, 
    MAP_TYPE_DICT, MAX_WALLS, MAX_SCORE)
import numpy as np
import random

class MapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_file=None, render=True, calc_score_on_step=True):
        self.map_view = Map(map_file=map_file, render=render)
        self.map_width = self.map_view.width
        self.map_height = self.map_view.height
        
        low = np.zeros(2, dtype=np.int32) # [0, 0]
        high = np.array([self.map_width-1, self.map_height-1], dtype=np.int32) # [map_width-1, map_height-1]
        self.action_space = spaces.Box(low, high, dtype=np.int32)
        self.observation_space = spaces.Dict({
            'map_type': spaces.Discrete(len(MAP_TYPE_DICT)),
            'num_checkpoints': spaces.Discrete(MAX_CHECKPOINTS),
            'num_teleports': spaces.Discrete(MAX_TELEPORTS),
            'num_walls': spaces.Discrete(MAX_WALLS),
            'teleports_used': spaces.Discrete(2*MAX_TELEPORTS),
            'walls_remaining': spaces.Discrete(MAX_WALLS),
            'score': spaces.Discrete(MAX_SCORE),
            'map': spaces.Box(0, len(TILE_DICT)-1, shape=MAP_BOUNDS)
        })
        self.initial_score = self.map_view.state['score']
        self.initial_walls = self.map_view.state['walls_remaining']
        self.input_layer = self.map_view.input_layer
        
        self.goal = self.map_view.state['score'] + self.map_view.state['walls_remaining']*2


    def step(self, action):
        old_score = int(self.map_view.state['score'])
        self.map_view.step(action=action)
        
        reward = self.map_view.state['score'] - old_score
        
        if self.map_view.state['score'] == 0:
            done = False
        elif self.map_view.state['walls_remaining'] == 0:
            done = True
        else:
            done = False
            
        info = {}
        
        return self.map_view.input_layer, reward, done, info

    def reset(self):
        return self.map_view.reset()

    def render(self, mode='human'):
        return


    #def close(self):

    def seed(self, seed=None):
        random.seed()
        return [seeding.np_random(seed)]
    
    # custom
    
    def get_one_hot(self, action):
        x = np.zeros((self.map_height,self.map_width))
        x[action[1], action[0]] = 1
        return x
        
    def get_random_action(self):
        if len(self.map_view.path_list) > 0:
            return random.choice(self.map_view.path_list)
        else:
            return self.action_space.sample()