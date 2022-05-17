
import numpy as np 
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
# from original_carracing import CarRacing
from new_carracing import CarRacing
from scipy.stats import skewnorm,norm

class PPOAgentTest:

    def __init__(self,checkpoints=None,viewer=None,model=None):
        self.env=CarRacing(checkpoints=checkpoints,viewer=viewer,use_ppo_agent=1)
        self.env._create_track()
        if model is None:
            self.model = PPO2.load('car_racing_weights.pkl')
        else:
            self.model = model

    def getRoadPoly(self):
        if self.env.road_poly is None or len(self.env.road_poly)==0:
            return None
        return self.env.road_poly

    def processReward(self,reward,hc):
        ppo_reward = skewnorm.pdf(reward, -4.5, 900, 250)/0.00285
        hc_reward=norm.pdf(hc, 3, 1.5)/0.325
        return 0.33*ppo_reward+0.67*hc_reward

    def test(self,show=False):
        #track generation failed penalty
        if self.env.road_poly is None or len(self.env.road_poly)==0:
            return -1
            
        hc=self.env.compute_curvature()
        self.env = DummyVecEnv([lambda :self.env])
        self.model.set_env(self.env)
        obs=self.env.reset()
        rewardsum = 0  

        for x in range(1000):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)
            rewardsum = rewardsum +rewards
            if show:
                self.env.render()
            if dones :
                self.env.close()
                break

        return self.processReward(rewardsum,hc)

        

if __name__ == "__main__":

    pid_env=PPOAgentTest()
    rewardsum=pid_env.test(True)
    # print(rewardsum)



