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

SCREEN_SIZE_X = 160
SCREEN_SIZE_Y = 90

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
                obs = obs/255
                obs = tf.image.resize(obs, [160,90])
                #print(obs)
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

            #np.savez_compressed(filename +'_'+str(u), obs=obs_sequence, action=action_sequence,
            #reward = reward_sequence, done = done_sequence)

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


    def _02_train_vae(self, new_model = False, num_episodes=73, timesteps=25, epochs = 1):
        vae = self.load_vae(new_model)
        M = timesteps
        N = num_episodes
        data, N = self.import_data(N,M)
        print(data.shape)
        #print(vae.encoder.summary())
        #print(vae.decoder.summary())
        for epoch in range(epochs):
            print('EPOCH', str(epoch))
            vae.save_weights('./VAE/weights/' + self.name +'.ckpt')
            vae.train(data, epochs = 1)
        vae.save_weights('./VAE/weights/' + self.name +'.ckpt')

    def load_vae(self, new_model):
        if self.vae != None:
            return self.vae
        vae = VAE.VAE()
        if not new_model:
            vae.set_weights('./VAE/weights/' + self.name +'.ckpt')
            print("Set weights successfully")
        self.vae = vae
        return vae

    def get_filelist(n_files, rollout_path):
        filelist = os.listdir(rollout_path)
        filelist = [x for x in filelist if x != '.DS_Store']
        filelist.sort()
        length_filelist = len(filelist)

        if length_filelist > n_files:
            filelist = filelist[:n_files]

        if length_filelist < n_files:
            n_files = length_filelist
        return filelist, n_files


    def import_data(self, n_files, M):
        rollout_path = self.path + '/rollouts/'
        filelist, n_files = WorldModel.get_filelist(n_files, rollout_path)

        data = np.zeros((M*n_files, SCREEN_SIZE_Y, SCREEN_SIZE_X, 3), dtype=np.float32)
        idx = 0
        file_count = 0

        DIR_NAME = rollout_path

        for file in filelist:
            try:
                new_data = np.load(DIR_NAME + file)['obs']/255
                # we put this here because the video data isn't generated the same way

                print(new_data.shape)
                #new_data = tf.image.resize(new_data, [64,64])
                #resizing should happen when recordin data
                #print(new_data.shape)
                data[idx:(idx + M), :, :, :] = new_data
                idx = idx + M
                file_count += 1

                if file_count%50==0:
                    print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, n_files, idx))
            except Exception as e:
                raise
                print(e)
                print('Skipped {}...'.format(file))


        print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, n_files, idx))

        return data, n_files

    def gen_vae_comparison_ims(self, n):
        data, n = self.import_data(n, 25)
        print(data.shape)
        vae = self.load_vae(False)
        z = vae.encoder.predict(data)
        mu = z[0]
        log_var = z[1]

        print(mu.shape)

        recon = vae.full_model.predict(data)
        recon_path = self.path + '/recon/'

        if os.path.exists(recon_path):
            print("Using existing directory: " + recon_path)
        else:
            os.mkdir(recon_path)
            print('Made new directory: ' + recon_path)
        
        for i in range(n*25):
            print("mu", mu[i])
            print("log_var", log_var[i])
            mpimg.imsave(recon_path+'obs_' +str(i)+".png", data[i])
            mpimg.imsave(recon_path+'recon_'+str(i)+".png", recon[i])


    def gen_random_ims(self, n):
        vae = self.load_vae(False)
        im_path = self.path + '/new_ims/'
        
        if os.path.exists(im_path):
            print("Using existing directory: " + im_path)
        else:
            os.mkdir(im_path)
            print('Made new directory: ' + im_path)
        
        z = tf.random.normal([n,32])
        gen = vae.decoder.predict(z)
        for i in range(n):
            mpimg.imsave(im_path + str(i) + '.png', gen[i])

#########################################################################################33


    def _03_generate_vae_rollouts(self, n_files = 73):
        vae_rollout_path = self.path+'/vae_rollouts/'
        if os.path.exists(vae_rollout_path):
            print("Using existing directory: " + vae_rollout_path)
        else:
            os.mkdir(vae_rollout_path)
            print('Made new directory: ' + vae_rollout_path)
        rollout_path = self.path + '/rollouts/'
        
        filelist, n_files = WorldModel.get_filelist(n_files, rollout_path)

        vae = self.load_vae(False)

        file_count = 0
        for file in filelist:
            try:
                data = np.load(rollout_path + file)
                obs = data['obs']/255
                #fucked up by scaling by 255 twice
                mu, log_var, _ = vae.encoder.predict(obs)
                # We save the mu and log_var so in MDRNN training we can generate a new z each time
                np.savez_compressed(vae_rollout_path  + file, mu=mu, log_var=log_var)
                
                file_count += 1

                if file_count % 20 == 0: 
                    print('Encoded {} / {} episodes'.format(file_count, n_files))
            except Exception as e:
                print(e)
                print('Skipped {}...'.format(file))
        print('Encoded {} / {} episodes'.format(file_count, n_files))

#################################################################################################

    def _04_train_mdrnn(self, restart = False, N=73, batch_size = 20, steps = 1):

        vae_rollout_path = self.path+'/vae_rollouts/'
        mdrnn = self.load_mdrnn(new_model = restart)

        filelist, N = WorldModel.get_filelist(N, vae_rollout_path)

        for step in range(steps):
            print('STEP' + str(step))

            #z , action, rew, done = random_batch(filelist, batch_size)
            z = self.random_batch(filelist, batch_size)
            rnn_in = z[:, :-1, :] #np.concatenate([z[:, :-1, :], action[:, :-1, :]], axis = 2)
            rnn_out = z[:, 1:, :]

            mdrnn.train(rnn_in, rnn_out)

            if step % 10 == 0:
                mdrnn.model.save_weights('./MDRNN/weights/' + self.name +'.ckpt')
                print("Saved weights")

        mdrnn.model.save_weights('./MDRNN/weights/' + self.name +'.ckpt')
        print("Saved weights")

    def load_mdrnn(self, new_model = False, in_dim = 32, out_dim = 32, lstm_units=256, n_mixes=5):
        if self.mdrnn != None:
            return self.mdrnn
        mdrnn = MDRNN.MDRNN(in_dim, out_dim, lstm_units, n_mixes)
        if not new_model:
            mdrnn.set_weights('./MDRNN/weights/' + self.name +'.ckpt')
            print("Set weights successfully")
        self.mdrnn = mdrnn
        return mdrnn

    def random_batch(self, filelist, batch_size):
        """open batch_size files from filelist"""
        N_data = len(filelist)
        start_index = random.randrange(N_data)
        if start_index + batch_size > N_data:
            wrap = (start_index + batch_size) % N_data
            indices = [i for i in range(wrap)] + [i for i in range(start_index, N_data)]
        else:
            indices = range(start_index, start_index+batch_size)
        #indices = np.random.permutation(N_data)[0:batch_size]
        assert len(indices) == batch_size

        z_list = []
        #action_list = []
        #rew_list = []
        #done_list = []

        for i in indices:
            try:
                new_data = np.load(self.path + '/vae_rollouts/' + filelist[i], allow_pickle=True)

                # this is the latent distribution
                mu = new_data['mu']
                log_var = new_data['log_var']
                #action = new_data['action']
                #reward = new_data['reward']
                #done = new_data['done']

                #reward = np.expand_dims(reward, axis=1)
                #done = np.expand_dims(done, axis=1)

                s = log_var.shape

                z = mu + np.exp(log_var/2.0) * np.random.randn(*s)

                z_list.append(z)
                #action_list.append(action)
                #rew_list.append(reward)
                #done_list.append(done)

            except Exception as e:
                print(e)

        z_list = np.array(z_list)
        #action_list = np.array(action_list)
        #rew_list = np.array(rew_list)
        #done_list = np.array(done_list)

        return z_list #, action_list, rew_list, done_list



    def _05_play_dream(self):
        pass



    def load_controller(self):
        pass
