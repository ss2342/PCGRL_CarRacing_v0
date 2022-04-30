
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
# from Monitor import Monitor
import numpy as np
from typing import Optional
import math
import matplotlib.pyplot as plt
from pid_heuristic_test import PidTest
import time


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
ZOOM = 2.0  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]
CHECKPOINTS=18
RADIAN_ACTIONS=5
RADIUS_ACTIONS=5
class TrackGenerationEnv(gym.Env, EzPickle):

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        #Action contains a control point and a movement:increase/decrease its radian/radius
        self.action_space = spaces.Box(
                low=np.array([0,0,0]), high=np.array([CHECKPOINTS-1,RADIAN_ACTIONS-1,RADIUS_ACTIONS-1]), dtype=np.uint8
            )
        #Each control point has two value(radian,radius)
        self.observation_space = spaces.Box(
                low=-TRACK_RAD, high=TRACK_RAD, shape=(CHECKPOINTS, 2), dtype=np.float32
            )
        self.checkpoints=[]
        self.viewer = None
        self.screen = None
        self.clock = None
       
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
                rad = 1 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1 * TRACK_RAD

            checkpoints.append([alpha, rad])
        return checkpoints
 


    def step(self, action):
        reward=0
        done=False

        print(action)
        if action is not None:
            if action[1]==0:
                self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]+0.01
            if action[1]==1:
                self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]+0.05
            if action[1]==2:
                self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]-0.01
            if action[1]==3:
                self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]-0.05

            if action[2]==0:
                self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]+0.1
            if action[2]==1:
                self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]+0.5
            if action[2]==2:
                self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]-0.1
            if action[2]==3:
                self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]-0.5

            self.checkpoints.sort(key=lambda x:x[0])

        pid_env=PidTest()
        self.road_poly=pid_env.getRoadPoly()
        # reward=pid_env.test() 

        return self.checkpoints, reward, done, {}

    def render(self, mode: str = "human"):
        import pygame

        pygame.font.init()

        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

        zoom=1.5
        angle=0
        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))
        trans = (0 , 0)

        self._render_road(zoom, trans, angle)
        self.surf = pygame.transform.flip(self.surf, False, True)

        if mode == "human":
            pygame.event.pump()
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

        # if mode == "rgb_array":
        #     return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        # elif mode == "state_pixels":
        #     return self._create_image_array(self.surf, (STATE_W, STATE_H))
        # else:
        #     return self.isopen
    
    def reset(
        self,
        *,
        seed: Optional[int] = None):
        super().reset(seed=seed)
        self.checkpoints=self._checkpoints_generation()
        return self.step(None)[0]

    def _draw_colored_polygon(self, surface, poly, color, zoom, translation, angle):
        import pygame
        from pygame import gfxdraw

        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        gfxdraw.aapolygon(self.surf, poly, color)
        gfxdraw.filled_polygon(self.surf, poly, color)


    def _render_road(self, zoom, translation, angle):

        bounds = PLAYFIELD
        field = [
            (2 * bounds, 2 * bounds),
            (2 * bounds, 0),
            (0, 0),
            (0, 2 * bounds),
            ]
        self._draw_colored_polygon(
            self.surf, field, (102, 204, 102), zoom, translation, angle
        )

        k = bounds / (20.0)
        grass = []
        for x in range(0, 40, 2):
            for y in range(0, 40, 2):
                grass.append(
                    [
                        (k * x + k, k * y + 0),
                        (k * x + 0, k * y + 0),
                        (k * x + 0, k * y + k),
                        (k * x + k, k * y + k),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, (102, 230, 102), zoom, translation, angle
            )
        for poly,color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0] + PLAYFIELD, p[1] + PLAYFIELD) for p in poly]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
  

    

def test(iter):
    env = TrackGenerationEnv()
    # env = Monitor(env, './video', force=True)
    # video = VideoRecorder(env, './video/training.mp4', True)
    env.reset()
    # print(env.checkpoints)
    for i in range(iter):
        state, reward, done, info=env.step(env.action_space.sample())
        # print(i,"!!!!!!!!",env.checkpoints)
        env.render()
        time.sleep(0.1)
        # video.capture_frame() 
    # video.close()
    # video.enabled = False
    env.close()


if __name__ == "__main__":
    test(1000)
