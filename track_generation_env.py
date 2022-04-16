import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import numpy as np
from typing import Optional
import math

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 0.3  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
CHECKPOINTS=18

class TrackGenerationEnv(gym.Env, EzPickle):

  def __init__(self):
    EzPickle.__init__(self)
    self.seed()
    #Action contains a control point and a movement:increase/decrease its radian/radius
    self.action_space = spaces.Tuple((spaces.Discrete(CHECKPOINTS), spaces.Discrete(4)))
    #Each control point has two value(radian,radius)
    self.observation_space = spaces.Box(low=0, high=3, shape=(CHECKPOINTS, 2), dtype=np.float32)
    self.checkpoints=[]
    
  def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]

  def _checkpoints_generation(self):
    # CHECKPOINTS = 12

        # Create checkpoints
    checkpoints = []
    for c in range(CHECKPOINTS):
      noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
      alpha = 2 * math.pi * c / CHECKPOINTS + noise
      rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

      if c == 0:
          alpha = 0
          rad = 1.5 * TRACK_RAD
      if c == CHECKPOINTS - 1:
          alpha = 2 * math.pi * c / CHECKPOINTS
          self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
          rad = 1.5 * TRACK_RAD

      checkpoints.append([alpha, rad * math.cos(alpha), rad * math.sin(alpha)])
    return checkpoints
 
  # def _checkpoints_generation(self):
  #   checkpoints = []

  #   for c in range(CHECKPOINTS):
  #     alpha = self.np_random.uniform(0,1)
  #     rad = self.np_random.uniform(1 / 3, 1.5)
  #     checkpoints.append([alpha, rad])

  #   return checkpoints

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
    # super().reset(seed=seed)
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
