import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import numpy as np
from typing import Optional
import math
import matplotlib.pyplot as plt


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
ZOOM = 2.7  # Camera zoom
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
 


  def step(self, action):
    reward=0
    done=False
    # print(action)
    # print(self.checkpoints[action[0]])
    if  action[1]==0: # increase radian by 
      self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]+50
    elif action[1]==1:
      self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]-50
    elif action[1]==2:
      self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]+50
    elif action[1]==3:
      self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]-50
    print(self.checkpoints[action[0]])
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


def plot_map(checkpoints):
  start_alpha = (checkpoints[0][0]+checkpoints[-1][0]-2 * math.pi)/2
  x, y, beta = 1.5 * TRACK_RAD, 0, 0
  dest_i = 0
  laps = 0
  track = []
  no_freeze = 2500
  visited_other_side = False
  while True:
      alpha = math.atan2(y, x)
      if visited_other_side and alpha > 0:
          laps += 1
          visited_other_side = False
      if alpha < 0:
          visited_other_side = True
          alpha += 2 * math.pi

      while True:  # Find destination from checkpoints
          failed = True

          while True:
              dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
              if alpha <= dest_alpha:
                  failed = False
                  break
              dest_i += 1
              if dest_i % len(checkpoints) == 0:
                  break

          if not failed:
              break

          alpha -= 2 * math.pi
          continue

      r1x = math.cos(beta)
      r1y = math.sin(beta)
      p1x = -r1y#-siny
      p1y = r1x#cosx
      dest_dx = dest_x - x  # vector towards destination
      dest_dy = dest_y - y
      # destination vector projected on rad:
      proj = r1x * dest_dx + r1y * dest_dy
      while beta - alpha > 1.5 * math.pi:
          beta -= 2 * math.pi
      while beta - alpha < -1.5 * math.pi:
          beta += 2 * math.pi
      prev_beta = beta
      proj *= SCALE
      if proj > 0.3:
          beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
      if proj < -0.3:
          beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
      x += p1x * TRACK_DETAIL_STEP
      y += p1y * TRACK_DETAIL_STEP
      track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
      if laps > 4:
          break
      no_freeze -= 1
      if no_freeze == 0:
          break

  # Find closed loop range i1..i2, first loop should be ignored, second is OK
  i1, i2 = -1, -1
  i = len(track)
  while True:
      i -= 1
      if i == 0:
          return False
 
      pass_through_start = (
          track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
      )
      if pass_through_start and i2 == -1:
          i2 = i
      elif pass_through_start and i1 == -1:
          i1 = i
          break

  assert i1 != -1
  assert i2 != -1

  track = track[i1 : i2 - 1]


  # Create tiles
  road_poly=[]
  for i in range(len(track)):
      alpha1, beta1, x1, y1 = track[i]
      alpha2, beta2, x2, y2 = track[i - 1]
      road1_l = (
          x1 - TRACK_WIDTH * math.cos(beta1),
          y1 - TRACK_WIDTH * math.sin(beta1),
      )
      road1_r = (
          x1 + TRACK_WIDTH * math.cos(beta1),
          y1 + TRACK_WIDTH * math.sin(beta1),
      )
      road2_l = (
          x2 - TRACK_WIDTH * math.cos(beta2),
          y2 - TRACK_WIDTH * math.sin(beta2),
      )
      road2_r = (
          x2 + TRACK_WIDTH * math.cos(beta2),
          y2 + TRACK_WIDTH * math.sin(beta2),
      )
      vertices = [road1_l, road1_r, road2_r, road2_l]

      road_poly.append((road1_l, road1_r, road2_r, road2_l))


  x=[i[1]for i in checkpoints]
  y=[i[2]for i in checkpoints]
  xs=[i[2] for i in track]
  ys=[i[3] for i in track]

  road1_lx=[i[0][0]for i in road_poly]
  road1_ly=[i[0][1]for i in road_poly]

  road1_rx=[i[1][0]for i in road_poly]
  road1_ry=[i[1][1]for i in road_poly]

  road2_lx=[i[2][0]for i in road_poly]
  road2_ly=[i[2][1]for i in road_poly]

  road2_rx=[i[3][0]for i in road_poly]
  road2_ry=[i[3][1]for i in road_poly]
  plt.plot(road1_lx,road1_ly,label='road1_l')
  plt.plot(road1_rx,road1_ry,label='road1_r')

  plt.plot(road2_lx,road2_ly,label='road2_l')
  plt.plot(road2_rx,road2_ry,label='road2_r')

  # plt.legend()
  plt.plot(xs,ys)
  plt.plot(x, y, "o")
  for a,b in zip(x, y): 
      plt.text(a, b, str(round(a, 2))+', '+str(round(b, 2)))
  plt.title("Track")
  plt.show()
  # print(road_poly[0])

if __name__ == "__main__":
  env = TrackGenerationEnv()
  env.reset()
  old = env.checkpoints
  # print(old)
  # plot_map(old)
  env.step(env.action_space.sample())
  new = env.checkpoints
  # plot_map(new)
  # print(new)




