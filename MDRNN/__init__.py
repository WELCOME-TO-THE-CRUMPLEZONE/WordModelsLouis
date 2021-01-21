import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import mdn

LEARNING_RATE = 0.001


class MDRNN():
    def __init__(self, in_dim, out_dim, lstm_units, n_mixes):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lstm_units = lstm_units
        self.n_mixes = n_mixes

        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]

    def _build(self):
        
        # for training
        rnn_x = Input(shape = (None, self.in_dim))
        lstm = LSTM(self.lstm_units, return_sequences=True, return_state=True) #old?

        lstm_out, _, _ = lstm(rnn_x)
        mdn_layer = mdn.MDN(self.out_dim, self.n_mixes)

        mdn_model = mdn_layer(lstm_out)

        model = Model(rnn_x, mdn_model)

        # for prediction
        lstm_in_h = Input(shape=(self.lstm_units,))
        lstm_in_c = Input(shape=(self.lstm_units,))

        lstm_out_forward, lstm_out_h, lstm_out_c = lstm(rnn_x, initial_state = [lstm_in_h, lstm_in_c])

        mdn_forward = mdn_layer(lstm_out_forward)
        
        forward = Model([rnn_x] + [lstm_in_h, lstm_in_c], [mdn_forward, lstm_out_h, lstm_out_c])

        def rnn_loss(z_true, z_pred):
            assert z_true.shape[-1] == self.out_dim
            assert z_pred.shape[-1] == (2*self.out_dim + 1)*self.n_mixes

            z_loss = mdn.get_mixture_loss_func(self.out_dim, self.n_mixes)(z_true, z_pred)
            return z_loss

        opti = Adam(lr=LEARNING_RATE)
        model.compile(loss=rnn_loss, optimizer=opti)

        return (model, forward)

    def train(self, rnn_in, rnn_out):
        self.model.fit(rnn_in, rnn_out,
                shuffle=False,
                epochs=1,
                batch_size=len(rnn_in))

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def forward_sample(self, z_in, lstm_in_h, lstm_in_c):
        mdn_out, lstm_out_h, lstm_out_c = self.forward([z_in, lstm_in_h, lstm_in_c])
        z_sample = mdn.get_mixture_sampling_fun(self.out_dim, self.n_mixes)(mdn_out)
        return (z_sample, lstm_out_h, lstm_out_c)
