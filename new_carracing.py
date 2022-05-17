import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import seeding, EzPickle 
import matplotlib.pyplot as plt

import pyglet

pyglet.options["debug_gl"] = False
from pyglet import gl
# from track_generation_env import TrackGenerationEnv

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

CHECKPOINTS = 18


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)




class CarRacing(gym.Env, EzPickle):
    """
    ### Description
    The easiest continuous control task to learn from pixels - a top-down
    racing environment. Discrete control is reasonable in this environment as
    well; on/off discretization is fine.
    The game is solved when the agent consistently gets 900+ points.
    The generated track is random every episode.
    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gym/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.
    ### Action Space
    There are 3 actions: steering (-1 is full left, +1 is full right), gas,
    and breaking.
    ### Observation Space
    State consists of 96x96 pixels.
    ### Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited,
    where N is the total number of tiles visited in the track. For example,
    if you have finished in 732 frames, your reward is
    1000 - 0.1*732 = 926.8 points.
    ### Starting State
    The car starts at rest in the center of the road.
    ### Episode Termination
    The episode finishes when all of the tiles are visited. The car can also go
    outside of the playfield - that is, far off the track, in which case it will
    receive -100 reward and die.
    ### Arguments
    `lap_complete_percent` dictates the percentage of tiles that must be visited by
    the agent before a lap is considered complete.
    Passing `domain_randomize=True` enabled the domain randomized variant of the environment.
    In this scenario, the background and track colours are different on every reset.
    ### Version History
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version
    ### References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.
    ### Credits
    Created by Oleg Klimov
    """
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": FPS,
    }

    def __init__(self,checkpoints=None,viewer=None,use_ppo_agent=0,verbose=1,):
        EzPickle.__init__(self)
        self.checkpoints=checkpoints
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = viewer
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.road_poly = []
        self.reward = 0.0
        self.prev_reward = 0.0
        self.use_ppo_agent=use_ppo_agent
        self.frames_per_state=1
        self.grayscale=0
        self.verbose = verbose
        self.show_info_panel=True
        self.discretize_actions=None
        self.possible_hard_actions = ("NOTHING", "LEFT", "RIGHT", "ACCELERATE", "BREAK")
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        state_shape=[STATE_H, STATE_W]
        if self.use_ppo_agent == 1:
            self.discretize_actions = "hard"
            self.show_info_panel=False
            self.grayscale=1
            self.frames_per_state=4
            state_shape.append(self.frames_per_state)
            lst = list(range(self.frames_per_state))
            self._update_index = [lst[-1]] + lst[:-1]
        else:
            state_shape.append(3)
            
        state_shape=tuple(state_shape)

        self.observation_space = spaces.Box(low=0, high=255, shape=state_shape,dtype=np.uint8)

        if self.discretize_actions == "hard":
            self.action_space = spaces.Discrete(len(self.possible_hard_actions))
        else:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if len(self.road_poly) !=0:
            return
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        # if self.car is not None:
        self.car.destroy()

    # def _init_colors(self):
    #     if self.domain_randomize:
    #         # domain randomize the bg and grass colour
    #         self.road_color = self.np_random.uniform(0, 210, size=3)

    #         self.bg_color = self.np_random.uniform(0, 210, size=3)

    #         self.grass_color = np.copy(self.bg_color)
    #         idx = self.np_random.integers(3)
    #         self.grass_color[idx] += 20
    #     else:
    #         # default colours
    #         self.road_color = np.array([102, 102, 102])
    #         self.bg_color = np.array([102, 204, 102])
    #         self.grass_color = np.array([102, 230, 102])

    def _create_track(self):

        if self.checkpoints != None:
            checkpoints = self.checkpoints

        else:
        # Create checkpoints
            checkpoints = []
            for c in range(CHECKPOINTS):
                noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
                alpha = 2 * math.pi * c / CHECKPOINTS + noise
                rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

                if c == 0:
                    alpha = 0
                    rad = TRACK_RAD
                if c == CHECKPOINTS - 1:
                    alpha = 2 * math.pi * c / CHECKPOINTS
                    rad = TRACK_RAD

                checkpoints.append((alpha, rad))
        self.road = []

        # Go from one checkpoint to another to create track
        self.start_alpha = (+checkpoints[-1][0]-2 * math.pi)/2
        alpha=checkpoints[0][0]
        rad=checkpoints[0][1]
        x, y, beta = rad*math.cos(alpha), rad*math.sin(alpha), alpha
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
                    dest_alpha,dest_rad = checkpoints[dest_i % len(checkpoints)]
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
            p1x = -r1y
            p1y = r1x
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
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        # if self.verbose:
            # print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # # Red-white border on hard turns
        # border = [False] * len(track)
        # for i in range(len(track)):
        #     good = True
        #     oneside = 0
        #     for neg in range(BORDER_MIN_COUNT):
        #         beta1 = track[i - neg - 0][1]
        #         beta2 = track[i - neg - 1][1]
        #         good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
        #         oneside += np.sign(beta1 - beta2)
        #     good &= abs(oneside) == BORDER_MIN_COUNT
        #     border[i] = good
        # for i in range(len(track)):
        #     for neg in range(BORDER_MIN_COUNT):
        #         border[i - neg] |= border[i]

        # Create tiles
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
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.reward = 0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            # if border[i]:
            #     side = np.sign(beta2 - beta1)
            #     b1_l = (
            #         x1 + side * TRACK_WIDTH * math.cos(beta1),
            #         y1 + side * TRACK_WIDTH * math.sin(beta1),
            #     )
            #     b1_r = (
            #         x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
            #         y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
            #     )
            #     b2_l = (
            #         x2 + side * TRACK_WIDTH * math.cos(beta2),
            #         y2 + side * TRACK_WIDTH * math.sin(beta2),
            #     )
            #     b2_r = (
            #         x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
            #         y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
            #     )
            #     self.road_poly.append(
            #         (
            #             [b1_l, b1_r, b2_r, b2_l],
            #             (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
            #         )
            #     )
        self.track = track
        return True

    def compute_curvature(self):
        #first derivatives 
        dx = np.gradient([t[2] for t in self.track])
        dy = np.gradient([t[3] for t in self.track])

        #second derivatives 
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        #calculation of curvature from the typical formula
        curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5

        self.curvature=curvature
        self.curvature_sum(20,15)
        return self.cross_entropy()
        # self.bins,_=np.histogram(curvature,bins=16,range=(0,0.096))

    def curvature_sum(self,range,interval):
        curvature_bin=[]
        curv_len=len(self.curvature)
        start=0
        while start<curv_len:
            if start+range-1<curv_len:
                curvature_bin.append(np.sum(self.curvature[start:start+range-1]))
            else:
                curvature_bin.append(np.sum(self.curvature[start:curv_len-1]))
            start=start+interval

        self.curvature_bin,_=np.histogram(curvature_bin,bins=16,range=(0,0.96))

    def cross_entropy(self):
        pro=self.curvature_bin/np.sum(self.curvature_bin)
        pro=-pro*np.log2(pro)
        pro=np.nan_to_num(pro)
        CE=np.sum(pro)
        return CE
 

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.state = np.zeros(self.observation_space.shape)

        if len(self.road_poly)==0:
            while True:
                success = self._create_track()
                if success:
                    break
                if self.verbose:
                    print(
                        "retry to generate track (normal if there are not many"
                        "instances of this message)"
                    )
                return None
        self.car = Car(self.world, *self.track[0][1:4])
        for _ in range(self.frames_per_state+20): 
            obs = self.step(None)[0]

        return obs

    def _update_state(self,new_frame):
        if self.frames_per_state > 1:
            self.state[:,:,-1] = new_frame
            self.state = self.state[:,:,self._update_index]
        else:
            self.state = new_frame


    def _transform_action(self, action):
        if self.discretize_actions == "hard":
            # ("NOTHING", "LEFT", "RIGHT", "ACCELERATE", "BREAK")
            # angle, gas, break
            if action == 0: action = [ 0, 0, 0.0] # Nothing
            if action == 1: action = [-1, 0, 0.0] # Left
            if action == 2: action = [+1, 0, 0.0] # Right
            if action == 3: action = [ 0,+1, 0.0] # Accelerate
            if action == 4: action = [ 0, 0, 0.8] # break

        return action

    def step(self, action: np.ndarray):
        action = self._transform_action(action)
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self._update_state(self.render("state_pixels"))

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, {}

    def render(self, mode: str = "human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)

        if "score_label" not in self.__dict__:
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
        
        if "transform" not in self.__dict__:
            from gym.envs.classic_control import rendering
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        if self.show_info_panel:
            self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        if self.grayscale and mode !="rgb_array":
            arr_bw = np.dot(arr[...,:3], [0.299, 0.587, 0.114])
            arr = arr_bw
        return arr


    def render_road(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons_ = [
            +PLAYFIELD,
            +PLAYFIELD,
            0,
            +PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            +PLAYFIELD,
            0,
        ]

        k = PLAYFIELD / 20.0
        colors.extend([0.4, 0.9, 0.4, 1.0] * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend(
                    [
                        k * x + k,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + k,
                        0,
                        k * x + k,
                        k * y + k,
                        0,
                    ]
                )

        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)
        )  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()

    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    place * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h,
                    0,
                    (place + 0) * s,
                    h,
                    0,
                ]
            )

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    (place + 0) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    2 * h,
                    0,
                    (place + 0) * s,
                    2 * h,
                    0,
                ]
            )

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)
        )  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

    def close(self):
        if self.viewer is not None:
            # self.viewer.close()
            self.viewer = None

    def getRoadPoly(self):
        return self.road_poly


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CarRacing(use_ppo_agent=0)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor 

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        env.compute_curvature()
        # plt.hist(env.curvature_bin,bins=16,range=(0,0.96))
        # plt.show()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()