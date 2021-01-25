import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


### Default 64 x 64
INPUT_DIM = (64,64,3)
#INPUT_DIM = [140,107,3]

N_LAYERS = 4

CONV_FILTERS = [32,64,64, 128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']
CONV_T_PADDING = [0,0,0,0]

Z_DIM = 32

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5


## 160 x 90 small
#INPUT_DIM = [90,160,3]

#N_LAYERS = 4

#CONV_FILTERS = [32,64,64, 128]
#CONV_KERNEL_SIZES = [4,4,4,4]
#CONV_STRIDES = [2,2,2,2]
#CONV_ACTIVATIONS = ['relu','relu','relu','relu']

#DENSE_SIZE = 1024

#PROBLEM: THIS DECONV STUFF TAKES US TO A PARTICULAR DIMENSION
#CONV_T_FILTERS = [64,64,32,3]
#CONV_T_KERNEL_SIZES = [5,5,6,6]
#CONV_T_STRIDES = [2,2,2,2]
#CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

#Z_DIM = 32

#BATCH_SIZE = 32
#LEARNING_RATE = 0.0001
#KL_TOLERANCE = 0.5

# 160 x 90 deeper
#INPUT_DIM = [90,160,3]

#N_LAYERS = 6

#CONV_FILTERS = [32,32,32,64,64, 128]
#CONV_KERNEL_SIZES = [4,4,4,3,3,3]
#CONV_STRIDES = [2,2,2,1,1,1]
#CONV_ACTIVATIONS = ['relu','relu','relu','relu','relu','relu']

#DENSE_SIZE = 1024

#CONV_T_FILTERS = [64,64,32,32,32,3]
#CONV_T_KERNEL_SIZES = [(2,3),(2,4),(3,4),(4,4),(5,4),(6,6)]
#CONV_T_STRIDES = [2,2,2,2,2,2]
#CONV_T_ACTIVATIONS = ['relu','relu', 'relu','relu','relu','sigmoid']
#CONV_T_PADDING = [0,0,0,0,0,0]

#Z_DIM = 32

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
KL_TOLERANCE = 0.5



class Sampling(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon

class VAEModel(Model):
    """Represents the entire model"""
    def __init__(self, encoder, decoder, r_loss_factor, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.r_loss_factor = r_loss_factor

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.square(data - reconstruction), axis = [1,2,3]
            )
            reconstruction_loss *= self.r_loss_factor
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_sum(kl_loss, axis = 1)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
    def call(self,inputs):
        latent = self.encoder(inputs)
        return self.decoder(latent)



class VAE():
    """Stores the encoder, decoder, and full model separately"""
    def __init__(self):
        self.models = self._build()
        self.full_model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM
        self.learning_rate = LEARNING_RATE
        self.kl_tolerance = KL_TOLERANCE

    def _build(self):
        # Encoder
        vae_x = Input(shape=INPUT_DIM, name='observation_input')
        vae_e_layers = [vae_x]

        for i in range(N_LAYERS):
            vae_e = Conv2D(filters = CONV_FILTERS[i], kernel_size = CONV_KERNEL_SIZES[i], strides = CONV_STRIDES[i], activation = CONV_ACTIVATIONS[i], name = 'vae_c'+str(i))(vae_e_layers[i])
            vae_e_layers.append(vae_e)
            #print(vae_e)

        vae_z_in = Flatten()(vae_e_layers[-1])

        vae_z_mean = Dense(Z_DIM, name='mu')(vae_z_in)
        vae_z_log_var = Dense(Z_DIM, name='log_var')(vae_z_in)

        vae_z = Sampling(name='z')([vae_z_mean, vae_z_log_var])

        # Decoder
        vae_z_input = Input(shape=(Z_DIM,), name='z_input')

        vae_dense = Dense(1024, name='dense_layer')(vae_z_input)
        #print(vae_dense.shape)
        vae_unflatten = Reshape((1,1,DENSE_SIZE), name='unflatten')(vae_dense)
        #print(vae_unflatten.shape)

        vae_d_layers = [vae_unflatten]
        for i in range(N_LAYERS):
            vae_d = Conv2DTranspose(filters = CONV_T_FILTERS[i], kernel_size = CONV_T_KERNEL_SIZES[i], strides = CONV_T_STRIDES[i], activation=CONV_T_ACTIVATIONS[i], output_padding = CONV_T_PADDING[i], name='deconv_layer_'+str(i))(vae_d_layers[i])
            vae_d_layers.append(vae_d)

        #### MODELS

    
        vae_encoder = Model(vae_x, [vae_z_mean, vae_z_log_var, vae_z], name = 'encoder')
        vae_decoder = Model(vae_z_input, vae_d_layers[-1], name = 'decoder')

        vae_full = VAEModel(vae_encoder, vae_decoder, 10000)

        opti = Adam(lr=LEARNING_RATE)
        vae_full.compile(optimizer=opti)
        
        return (vae_full,vae_encoder, vae_decoder)

    def set_weights(self, filepath):
        self.full_model.load_weights(filepath)

    def train(self, data):
        #print(data.shape)
        self.full_model.fit(data, data,
                shuffle=True,
                epochs=1,
                batch_size=BATCH_SIZE)
        
    def save_weights(self, filepath):
        self.full_model.save_weights(filepath)
