import tensorflow as tf
from singleoutput_model import Singleoutput
from sklearn.metrics import mean_absolute_error
import numpy as np


class DNN_Single(Singleoutput):
    def __init__(self, X_train, y_train, X_test, y_test, target):
        super(DNN_Single, self).__init__(X_train, y_train, X_test, y_test, target, None)

    def set_model(self, model):
        self.model = model

    def train_model(self):
        inp = tf.keras.layers.Input(shape=(99,))
        x = tf.keras.layers.Dense(25, activation='relu')(inp)

        x = tf.keras.layers.Dropout(.2, input_shape=(x.shape))(x)

        s1 = tf.keras.layers.Dense(5, activation='relu')(x)

        s2 = tf.keras.layers.Dense(5, activation='relu')(x)

        s3 = tf.keras.layers.concatenate([s1, s2])

        s1 = tf.keras.layers.Dense(1, activation='linear')(s3)

        out = s1

        nn_improve_2 = tf.keras.Model(inputs=inp, outputs=out, name='improve_2')

        nn_improve_2.compile(optimizer='adam', loss='mae', metrics='mae')

        nn_improve_2.fit(self.X_train, self.y_train, batch_size=10, epochs=100, verbose=0)

        self.model = nn_improve_2


    def get_model_plot(self, png_name='model'):
        tf.keras.utils.plot_model(self.model, f"{png_name}.png", show_shapes=True)

    def get_summary(self):
        return self.model.summary()

    def save_mode(self, model_name_location):
        self.model.save(model_name_location.split('.')[0])
