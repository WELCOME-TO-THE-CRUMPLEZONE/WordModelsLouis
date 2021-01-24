import numpy as np
import tensorflow as tf
import gym
import retro
import random
import os

import MDRNN
import VAE
import config as c

import matplotlib.image as mpimg

SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    
    def load(name):
        """Load the settings of an existing WordModel"""
        settings_path  = './data/'+ self.name + '/settings'
        if os.path.exists(self.path):
            print("Using existing directory: " + self.path)
        else:
            print('Tried to load non-existent settings', self.path)

#################################### 01 Generate Rollouts #####################################

    def _01_generate_rollouts(self, total_episodes, time_steps, file_lengths, action_refresh_rate, render):
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
            filename = rollout_path + str(episode_id)

            obs = env.reset()
            done = False
            reward = 0

            t = 0
            u = 0
            obs_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []


            while t < time_steps and not done:
                if t % action_refresh_rate == 0:
                    action = self.get_action(t, env)

                # obs = config.adjust_obs
                obs = tf.image.resize(obs, [140,107])
                #print(obs.shape)
                #obs = obs.numpy()/255
                #mpimg.imsave('./data/ims/'+str(t)+".png", obs)
                obs_sequence.append(obs)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)

                obs, reward, done, info = env.step(action)

                t = t + 1

                if render:
                    env.render()

                if t % file_lengths == 0 and t != 0:
                    np.savez_compressed(filename +'_'+ str(u), obs=obs_sequence, action=action_sequence,
                            reward = reward_sequence, done = done_sequence)
                    u = u + 1
                    obs_sequence = []
                    action_sequence = []
                    reward_sequence = []
                    done_sequence = []

            print("Episode {} finished after {} timesteps".format(s, t))

            np.savez_compressed(filename +'_'+str(u), obs=obs_sequence, action=action_sequence,
                    reward = reward_sequence, done = done_sequence)

            s = s + 1
        env.close()

        return
    
    def load_env(self, env_name):
        if self.env != None:
            return self.env
        else:
            if env_name == "SimCity-Snes":
                retro.data.Integrations.add_custom_path(
                    os.path.join(SCRIPT_DIR, "custom_integrations")
                )
                print("SimCity-Snes" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
                env = retro.make("SimCity-Snes", inttype=retro.data.Integrations.ALL)
                return env

            env = gym.make(env_name)
            self.env = env
            return env
            

    def get_action(self, t, env): 
        return env.action_space.sample()

######################################## 02 Train VAE ##########################################


    def _02_train_vae(self, new_model = False):
        vae = self.load_vae(new_model)
        
        data, N = self.import_data(N,M)
        for epoch in range(epochs):
            print('EPOCH', str(epoch))
            vae.save_weights('./VAE/weights/' + self.name +'.h5')
            vae.train(data)
        vae.save_weights('./VAE/weights/' + self.name +'.h5')

    def load_vae(self, new_model):
        if self.vae != None:
            return self.vae
        vae = VAE.VAE()
        if not new_model:
            vae.set_weights('./VAE/weights/' + self.name +'.h5')
        self.vae = vae
        return vae

    def import_data(self, n_files, M):
        rollout_path = self.path + '/rollouts/'
        filelist = os.listdir()
        filelist = [x for x in filelist if x != '.DS_Store']
        filelist.sort()
        length_filelist = len(filelist)


        if length_filelist > n_files:
            filelist = filelist[:n_files]

        if length_filelist < n_files:
            n_files = length_filelist

        data = np.zeros((M*n_files, SCREEN_SIZE_X, SCREEN_SIZE_Y, 3), dtype=np.float32)
        idx = 0
        file_count = 0


        for file in filelist:
            try:
                new_data = np.load(DIR_NAME + file)['obs']
                #new_data = tf.image.resize(new_data, [64,64])
                #resizing should happen when recordin data
                data[idx:(idx + M), :, :, :] = new_data

                idx = idx + M
                file_count += 1

                if file_count%50==0:
                    print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
            except Exception as e:
                print(e)
                print('Skipped {}...'.format(file))

        print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))

        return data, N





    def _03_generate_vae_rollouts(self):
        pass

    def _04_train_mdrnn(self, restart = False):
        pass

    def _05_play_dream(self):
        pass



    def load_controller(self):
        pass
