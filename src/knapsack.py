import numpy as np
import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import copy
from typing import Optional

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")

class KnapsackEnv(gym.Env):
    '''
    Unbounded Knapsack Problem

    The Knapsack Problem (KP) is a combinatorial optimization problem which
    requires the user to select from a range of goods of different values and
    weights in order to maximize the value of the selected items within a 
    given weight limit. This version is unbounded meaning that we can select
    items without limit. 

    The episodes proceed by selecting items and placing them into the
    knapsack one at a time until the weight limit is reached or exceeded, at
    which point the episode ends.

    Observation:
        Type: Tuple, Discrete
        0: list of item weights
        1: list of item values
        2: maximum weight of the knapsack
        3: current weight in knapsack

    Actions:
        Type: Discrete
        0: Place item 0 into knapsack
        1: Place item 1 into knapsack
        2: ...

    Reward:
        Value of item successfully placed into knapsack or 0 if the item
        doesn't fit, at which point the episode ends.

    Starting State:
        Lists of available items and empty knapsack.

    Episode Termination:
        Full knapsack or selection that puts the knapsack over the limit.
    '''
    
    # Internal list of placed items for better rendering
    _collected_items = []
    
    def __init__(self, N=200, W=200, *args, **kwargs):
        # Generate data with consistent random seed to ensure reproducibility
        self.N = N
        self.max_weight = 150
        self.current_weight = 0
        self._max_reward = 10000
        self.seed = 0
        self.item_numbers = np.arange(self.N)
        self.item_weights = np.random.randint(1, 100, size=self.N)
        self.item_values = np.random.randint(0, 100, size=self.N)
        self.over_packed_penalty = 0
        self.randomize_params_on_reset = False
        self._collected_items.clear()
        # Add env_config, if any
        assign_env_config(self, kwargs)
        self.set_seed()

        obs_space = spaces.Box(
            0, self.max_weight, shape=(2*self.N + 1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(
                0, self.max_weight, shape=(2, self.N + 1), dtype=np.int32)
        
        self.reset()
        
    def step(self, item):
        # Check that item will fit
        if self.item_weights[item] + self.current_weight <= self.max_weight:
            self.current_weight += self.item_weights[item]
            reward = self.item_values[item]
            self._collected_items.append(item)
            if self.current_weight == self.max_weight:
                done = True
            else:
                done = False
        else:
            # End trial if over weight
            reward = self.over_packed_penalty
            done = True
            
        self._update_state()
        return self.state, reward, done, False, {'current_weight': self.current_weight}
    
    def _get_obs(self):
        return self.state
    
    def _update_state(self):
        state = np.vstack([
            self.item_weights,
            self.item_values], dtype=np.int32)
        self.state = np.hstack([
            state,
            np.array([
                [self.max_weight],
                 [self.current_weight]])
            ], dtype=np.int32)        
    
    def sample_action(self):
        return np.random.choice(self.item_numbers)

    def set_seed(self, seed=None):
        if seed == None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if seed is not None:
            self.set_seed(seed)
        if self.randomize_params_on_reset:
            self.item_weights = np.random.randint(1, 100, size=self.N)
            self.item_values = np.random.randint(0, 100, size=self.N)
        self.current_weight = 0
        self._collected_items.clear()
        self._update_state()
        return self.state, {}
        
    def render(self):
        total_value = 0
        total_weight = 0
        for i in range(self.N) :
            if i in self._collected_items :
                total_value += self.item_values[i]
                total_weight += self.item_weights[i]
        print(self._collected_items, total_value, total_weight)
        
        # RlLib requirement: Make sure you either return a uint8/w x h x 3 (RGB) image or handle rendering in a window and then return `True`.
        return True

