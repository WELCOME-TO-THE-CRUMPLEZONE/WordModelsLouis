import numpy as np
import gym
import random
import os

import MDRNN
import VAE


class WorldModel():
    """A class for running a complete world models experiment (minus training the final controller)"""
    def __init__(self, env_name, name = None):
        self.name = name
        if name == None:
            self.name = env_name
        
        self.path = './data/'+ self.name
        if os.path.exists(self.path):
            print("Using existing directory: " + self.path)
        else:
            os.mkdir(self.path)
            print('Made new directory: ' + self.path)

        self.env_name = env_name
        self.env = None
        self.vae = None
        self.mdrnn = None
    

    def _01_generate_rollouts(self, total_episodes, time_steps, action_refresh_rate, render):
        rollout_path = self.path + '/rollouts/'

        if os.path.exists(rollout_path):
            print("Using existing directory: " + rollout_path)
        else:
            os.mkdir(rollout_path)
            print('Made new directory: ' + rollout_path)

        env = self.load_env(self.env_name)

        print("Generating data for env " + self.env_name)
        s = 0
        while s < total_episodes:
            episode_id = random.randint(0, 2**31-1)
            filename = rollout_path + str(episode_id) + '.npz'

            obs = env.reset()
            done = False
            reward = 0

            t = 0
            obs_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []


            while t < time_steps and not done:
                if t % action_refresh_rate == 0:
                    action = self.get_action(t, env)

                # obs = config.adjust_obs
                obs_sequence.append(obs)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)

                obs, reward, done, info = env.step(action)

                t = t + 1

                if render:
                    env.render()
            print("Episode {} finished after {} timesteps".format(s, t))

            np.savez_compressed(filename, obs=obs_sequence, action=action_sequence,
                    reward = reward_sequence, done = done_sequence)

            s = s + 1
        env.close()

        return
    
    def load_env(self, env_name):
        if self.env != None:
            return self.env
        else:
            env = gym.make(env_name)
            return env
            

    def get_action(self, t, env): 
        return env.action_space.sample()

    def _02_train_vae(self, restart = False):
        if self.vae == None:
            self.vae = load_vae()

    def _03_generate_vae_rollouts(self):
        pass

    def _04_train_mdrnn(self, restart = False):
        pass

    def _05_play_dream(self):
        pass



    def load_controller(self):
        pass
