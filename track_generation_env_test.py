
import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import math
import matplotlib.pyplot as plt
from pid_heuristic_test import PidTest
from ppo_agent import PPOAgentTest
from stable_baselines import PPO2
import time
import pyglet
pyglet.options["debug_gl"] = False
from pyglet import gl
import datetime
import tensorflow as tf

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
        #Action contains a control point and a movement
        # self.action_space = spaces.Box(
        #         low=np.array([0,0,0]), high=np.array([CHECKPOINTS-1,RADIAN_ACTIONS-1,RADIUS_ACTIONS-1]), dtype=np.uint8
        #     )
        # self.action_space= spaces.Discrete(CHECKPOINTS)
        self.action_space = spaces.Box(
                low=-0.1, high=+0.1, shape=(CHECKPOINTS*2,), dtype=np.float32
            )

        #Each control point has two value(radian,radius)
        self.observation_space = spaces.Box(
                low=-PLAYFIELD, high=PLAYFIELD, shape=(CHECKPOINTS, 2), dtype=np.float32
            )
        self.checkpoints=[]
        self.viewer = None
        self.road_poly= None
        self.model=None
        self.tiles_num=0
        self.full_reward_lst = []
        # self.reward_lst = []
        # self.full_reward_lst.append(self.reward_lst)

    def set_ppo_model(self,model):
        self.model = model
       
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
                rad = 1.5 * TRACK_RAD

            checkpoints.append([alpha, rad])
        return checkpoints
 


    def step(self, action):
        reward=0
        done=False
        
        if action is not None:
            for i in range(CHECKPOINTS):
                self.checkpoints[i][0]=self.checkpoints[i][0]+action[i*2]
                self.checkpoints[i][1]=self.checkpoints[i][1]+action[i*2+1]
        # if action is not None:
        #     action=action.astype(np.uint8)
        #     if action[1]==0:
        #         self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]+0.01
        #     if action[1]==1:
        #         self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]+0.1
        #     if action[1]==2:
        #         self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]-0.01
        #     if action[1]==3:
        #         self.checkpoints[action[0]][0]=self.checkpoints[action[0]][0]-0.1

        #     if action[2]==0:
        #         self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]+0.1
        #     if action[2]==1:
        #         self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]+1
        #     if action[2]==2:
        #         self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]-0.1
        #     if action[2]==3:
        #         self.checkpoints[action[0]][1]=self.checkpoints[action[0]][1]-1

        #     self.checkpoints.sort(key=lambda x:x[0])
        
        
        # self.checkpoints=action
        for c in self.checkpoints:
            while c[0]>2 * math.pi:
                c[0]=c[0]-2 * math.pi
            while c[0]<0:
                c[0]=c[0]+2 * math.pi
        self.checkpoints.sort(key=lambda x:x[0])
        test_env=PPOAgentTest(self.checkpoints,self.viewer,self.model)

        #failed to generate track, use last road poly
        if test_env.getRoadPoly() is not None:
            self.road_poly=test_env.getRoadPoly()

        reward=test_env.test()

        if reward==-1: #new track generated
            self.tiles_num=0
            while 1:
                self.checkpoints=self._checkpoints_generation()
                test_env=PPOAgentTest(self.checkpoints,self.viewer,self.model)
                if test_env.getRoadPoly() is not None:
                    self.road_poly=test_env.getRoadPoly()
                    break
        else:
            tiles_num=len(self.road_poly)
            if tiles_num<0.8*self.tiles_num:
                # print("decrease too much tiles -0.5")
                reward=reward-0.5
            self.tiles_num=tiles_num
        
        with open("rewards.txt", "a") as o:
            if reward != -1:
                print(reward[0], file=o)
            else:
                print(reward, file=o)
            o.close()
        # print(f'reward{reward}')
        
        # name = 'full'
        # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # save_folder = f"./pcgrl_checkpoints/{name}_{timestamp}"
        # summary_writer = tf.summary.SummaryWriter(save_folder)
        # checkpoint = tf.train.Checkpoint(
            # model=self.model)
        # checkpoint_prefix = save_folder + '/ckpt'
        # tf.summary.scalar('reward', reward)
        # self.reward_lst.append(reward)
        # print(f'full reward list: {self.full_reward_lst}')
        return self.checkpoints, reward, done, {}

    def render(self, mode: str = "human"):
        assert mode in ["human", "state_pixels", "rgb_array"]

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()


        zoom=1.5
        angle=0
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(WINDOW_W / 2,WINDOW_H / 2)
        self.transform.set_rotation(angle)

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

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def reset(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        self.tiles_num=0
        while 1:
            self.checkpoints=self._checkpoints_generation()
            self.step(None)
            if self.road_poly is not None:
                return self.checkpoints
            # print("retry")


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


    

def test(iter):
    env = TrackGenerationEnv()
    env.set_ppo_model(PPO2.load('car_racing_weights.pkl'))
    # env = Monitor(env, './video', force=True)
    # video = VideoRecorder(env, './video/training.mp4', True)
    env.reset()
    # print(env.checkpoints)
    for i in range(iter):
        state, reward, done, info=env.step(env.action_space.sample())
        pixel=env.render()
        # print(type(pixel))
        # plt.imshow(pixel)
        # plt.show()
        time.sleep(0.1)
        # video.capture_frame() 
    # video.close()
    # video.enabled = False
    env.close()


if __name__ == "__main__":
    test(1000)
