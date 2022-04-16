import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import Optional
import math

CHECKPOINTS=18

class TrackGenerationEnv(gym.Env):

  def __init__(self):
    #Action contains a control point and a movement:increase/decrease its radian/radius
    self.action_space = spaces.Tuple((spaces.Discrete(CHECKPOINTS), spaces.Discrete(4)))
    #Each control point has two value(radian,radius)
    self.observation_space = spaces.Box(low=0, high=3, shape=(CHECKPOINTS, 2), dtype=np.float32)
    self.checkpoints=[]

  def _checkpoints_generation(self):
    checkpoints = []

    for c in range(CHECKPOINTS):
      alpha = self.np_random.uniform(0,1)
      rad = self.np_random.uniform(1 / 3, 1.5)
      checkpoints.append([alpha, rad])

    return checkpoints

  def step(self, action):
    reward=0
    done=False
    print(action)
    if  action[1]==0:
      self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]+0.01
    elif action[1]==1:
      self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]-0.01
    elif action[1]==2:
      self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]+0.01
    elif action[1]==3:
      self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]-0.01  


    return self.checkpoints, reward, done, {}
    
  def reset(
      self,
      *,
      seed: Optional[int] = None):
    super().reset(seed=seed)
    self.checkpoints=self._checkpoints_generation()

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...

if __name__ == "__main__":
  env = TrackGenerationEnv()
  env.reset()
  print(env.checkpoints)
  env.step(env.action_space.sample())
  print(env.checkpoints)
