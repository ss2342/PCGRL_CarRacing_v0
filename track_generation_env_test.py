
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
# from Monitor import Monitor
import numpy as np
from typing import Optional
import math
import matplotlib.pyplot as plt
from pid_heuristic_test import pcgrl_input
import time

import pyglet
from pyglet import gl


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

def get_track(checkpoints):
    start_alpha = (+checkpoints[-1][0]-2 * math.pi)/2
    alpha=checkpoints[0][0]
    rad=checkpoints[0][1]
    x, y, beta = rad*math.cos(alpha), rad*math.cos(alpha), alpha
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
                dest_alpha, dest_rad = checkpoints[dest_i % len(checkpoints)]
                dest_x=dest_rad*math.cos(dest_alpha)
                dest_y=dest_rad*math.sin(dest_alpha)
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
        if laps > 5:
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

        road_poly.append((road1_l, road1_r, road2_r, road2_l))
    return road_poly

class TrackGenerationEnv(gym.Env, EzPickle):

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        #Action contains a control point and a movement:increase/decrease its radian/radius
        self.action_space = spaces.Tuple((spaces.Discrete(CHECKPOINTS), spaces.Box(low=np.array([-2*math.pi/50,-TRACK_RAD/50]),high=np.array([2*math.pi/50,TRACK_RAD/50]),dtype=np.float32)))
        #Each control point has two value(radian,radius)
        self.observation_space = spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
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

        # print(action)
        self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]+action[1][0]
        self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]+action[1][1]

        self.checkpoints.sort(key=lambda x:x[0])
        
        state=self.render("rgb_array")

        ##pid reward
        # reward=pcgrl_input(self.checkpoints)

        return state, reward, done, {}
    
    def reset(
        self,
        *,
        seed: Optional[int] = None):
        # super().reset(seed=seed)
        self.checkpoints=self._checkpoints_generation()

    def _draw_colored_polygon(self, surface, poly, color, zoom, translation, angle):
        import pygame
        from pygame import gfxdraw

        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        gfxdraw.aapolygon(self.surf, poly, color)
        gfxdraw.filled_polygon(self.surf, poly, color)


    def get_road_poly(self, checkpoints):
        road_poly = get_track(checkpoints)
        return road_poly

    def _render_road(self, zoom, translation, angle, checkpoints):

        road_poly = self.get_road_poly(checkpoints)
        bounds = PLAYFIELD
        field = [
            (2 * bounds, 2 * bounds),
            (2 * bounds, 0),
            (0, 0),
            (0, 2 * bounds),
            ]
        trans_field = []
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
        color=(102,102,102)
        for poly in road_poly:
            # converting to pixel coordinates
            poly = [(p[0] + PLAYFIELD, p[1] + PLAYFIELD) for p in poly]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
  
    def render(self, mode='human', close=False):

        import pygame
        

        pygame.font.init()

        assert mode in ["human", "state_pixels", "rgb_array"]

        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

        elif self.screen is None and mode == "rgb_array":
            pygame.init
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))


        
        if self.clock is None:
            self.clock = pygame.time.Clock()


        # road_poly = get_track(self.checkpoints)
        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))
        zoom = 0.7
        trans = (WINDOW_W / 2, WINDOW_H / 2)
        self._render_road(zoom, trans, 0, self.checkpoints)
        self.screen.blit(self.surf, (-WINDOW_W / 4, -WINDOW_H / 4))
        pygame.display.flip()

        # computing transformations
        
        # if self.viewer is None: 
        #     from gym.envs.classic_control import rendering

        #     self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
        #     self.transform = rendering.Transform()
        
        # self.transform.set_scale(ZOOM, ZOOM)
        # self.transform.set_translation(
        #     WINDOW_W / 2,
        #     WINDOW_H / 2,
        # )
        # self.transform.set_rotation(0)


        # win = self.viewer.window
        # win.switch_to()
        # win.dispatch_events()
        # win.clear()

        # self.transform.enable()
        # road_poly=create_track(self.checkpoints)

        # if mode == "rgb_array":
        #     VP_W = STATE_W
        #     VP_H = STATE_H
        # elif mode == "human":
        #     VP_W = WINDOW_W
        #     VP_H = WINDOW_H
        
        # gl.glViewport(0, 0, VP_W, VP_H)
        # self.render_road(road_poly)
        # self.transform.disable()
        # if mode == "human":
        #     win.flip()
        #     return self.viewer.isopen

        # image_data = (
        #     pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        # )
        # arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        # arr = arr.reshape(VP_H, VP_W, 4)
        # arr = arr[::-1, :, 0:3]

        # return arr

    # def render_road(self,road_poly):
    #     colors = [0.4, 0.8, 0.4, 1.0] * 4
    #     polygons_ = [
    #         +PLAYFIELD,
    #         +PLAYFIELD,
    #         0,
    #         +PLAYFIELD,
    #         -PLAYFIELD,
    #         0,
    #         -PLAYFIELD,
    #         -PLAYFIELD,
    #         0,
    #         -PLAYFIELD,
    #         +PLAYFIELD,
    #         0,
    #     ]

    #     k = PLAYFIELD / 20.0
    #     colors.extend([0.4, 0.9, 0.4, 1.0] * 4 * 20 * 20)
    #     for x in range(-20, 20, 2):
    #         for y in range(-20, 20, 2):
    #             polygons_.extend(
    #                 [
    #                     k * x + k,
    #                     k * y + 0,
    #                     0,
    #                     k * x + 0,
    #                     k * y + 0,
    #                     0,
    #                     k * x + 0,
    #                     k * y + k,
    #                     0,
    #                     k * x + k,
    #                     k * y + k,
    #                     0,
    #                 ]
    #             )
    #     color=ROAD_COLOR
    #     for poly in road_poly:
    #         colors.extend([color[0], color[1], color[2], 1] * len(poly))
    #         for p in poly:
    #             polygons_.extend([p[0], p[1], 0])

    #     vl = pyglet.graphics.vertex_list(
    #         len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)
    #     )  # gl.GL_QUADS,
    #     vl.draw(gl.GL_QUADS)
    #     vl.delete()

    






# def plot_map(checkpoints):
#   start_alpha = (checkpoints[0][0]+checkpoints[-1][0]-2 * math.pi)/2
#   x, y, beta = 1.5 * TRACK_RAD, 0, 0
#   dest_i = 0
#   laps = 0
#   track = []
#   no_freeze = 2500
#   visited_other_side = False
#   while True:
#       alpha = math.atan2(y, x)
#       if visited_other_side and alpha > 0:
#           laps += 1
#           visited_other_side = False
#       if alpha < 0:
#           visited_other_side = True
#           alpha += 2 * math.pi

#       while True:  # Find destination from checkpoints
#           failed = True

#           while True:
#               dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
#               if alpha <= dest_alpha:
#                   failed = False
#                   break
#               dest_i += 1
#               if dest_i % len(checkpoints) == 0:
#                   break

#           if not failed:
#               break

#           alpha -= 2 * math.pi
#           continue

#       r1x = math.cos(beta)
#       r1y = math.sin(beta)
#       p1x = -r1y#-siny
#       p1y = r1x#cosx
#       dest_dx = dest_x - x  # vector towards destination
#       dest_dy = dest_y - y
#       # destination vector projected on rad:
#       proj = r1x * dest_dx + r1y * dest_dy
#       while beta - alpha > 1.5 * math.pi:
#           beta -= 2 * math.pi
#       while beta - alpha < -1.5 * math.pi:
#           beta += 2 * math.pi
#       prev_beta = beta
#       proj *= SCALE
#       if proj > 0.3:
#           beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
#       if proj < -0.3:
#           beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
#       x += p1x * TRACK_DETAIL_STEP
#       y += p1y * TRACK_DETAIL_STEP
#       track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
#       if laps > 4:
#           break
#       no_freeze -= 1
#       if no_freeze == 0:
#           break

#   # Find closed loop range i1..i2, first loop should be ignored, second is OK
#   i1, i2 = -1, -1
#   i = len(track)
#   while True:
#       i -= 1
#       if i == 0:
#           return False
 
#       pass_through_start = (
#           track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
#       )
#       if pass_through_start and i2 == -1:
#           i2 = i
#       elif pass_through_start and i1 == -1:
#           i1 = i
#           break

#   assert i1 != -1
#   assert i2 != -1

#   track = track[i1 : i2 - 1]


#   # Create tiles
#   road_poly=[]
#   for i in range(len(track)):
#       alpha1, beta1, x1, y1 = track[i]
#       alpha2, beta2, x2, y2 = track[i - 1]
#       road1_l = (
#           x1 - TRACK_WIDTH * math.cos(beta1),
#           y1 - TRACK_WIDTH * math.sin(beta1),
#       )
#       road1_r = (
#           x1 + TRACK_WIDTH * math.cos(beta1),
#           y1 + TRACK_WIDTH * math.sin(beta1),
#       )
#       road2_l = (
#           x2 - TRACK_WIDTH * math.cos(beta2),
#           y2 - TRACK_WIDTH * math.sin(beta2),
#       )
#       road2_r = (
#           x2 + TRACK_WIDTH * math.cos(beta2),
#           y2 + TRACK_WIDTH * math.sin(beta2),
#       )
#       vertices = [road1_l, road1_r, road2_r, road2_l]

#       road_poly.append((road1_l, road1_r, road2_r, road2_l))


#   x=[i[1]for i in checkpoints]
#   y=[i[2]for i in checkpoints]
#   xs=[i[2] for i in track]
#   ys=[i[3] for i in track]

#   road1_lx=[i[0][0]for i in road_poly]
#   road1_ly=[i[0][1]for i in road_poly]

#   road1_rx=[i[1][0]for i in road_poly]
#   road1_ry=[i[1][1]for i in road_poly]

#   road2_lx=[i[2][0]for i in road_poly]
#   road2_ly=[i[2][1]for i in road_poly]

#   road2_rx=[i[3][0]for i in road_poly]
#   road2_ry=[i[3][1]for i in road_poly]
#   plt.plot(road1_lx,road1_ly,label='road1_l')
#   plt.plot(road1_rx,road1_ry,label='road1_r')

#   plt.plot(road2_lx,road2_ly,label='road2_l')
#   plt.plot(road2_rx,road2_ry,label='road2_r')

#   # plt.legend()
#   plt.plot(xs,ys)
#   plt.plot(x, y, "o")
#   for a,b in zip(x, y): 
#       plt.text(a, b, str(round(a, 2))+', '+str(round(b, 2)))
#   plt.title("Track")
#   plt.show()
#   # print(road_poly[0])

def test(iter):
    env = TrackGenerationEnv()
    # env = Monitor(env, './video', force=True)
    # video = VideoRecorder(env, './video/training.mp4', True)
    env.reset()
    # print(env.checkpoints)
    for i in range(iter):
        state, reward, done, info=env.step(env.action_space.sample())
        # print(env.checkpoints)
        env.render()
        time.sleep(0.1)
        # video.capture_frame()
    # video.close()
    # video.enabled = False
    env.close()


if __name__ == "__main__":
    test(1000)
