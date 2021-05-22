import tensorflow as tf
from multioutput_model import Multioutput
from sklearn.metrics import mean_absolute_error
import numpy as np


class DNN(Multioutput):
    def __init__(self, X_train, y_train, X_test, y_test):
        super(DNN, self).__init__(X_train, y_train, X_test, y_test, None)

    def set_model(self, model):
        self.model = model

    def train_model(self):
        inp = tf.keras.layers.Input(shape=(99,))
        x = tf.keras.layers.Dense(50, activation='relu')(inp)
        
        x = tf.keras.layers.Dropout(.2, input_shape=(x.shape))(x)

        s1 = tf.keras.layers.Dense(10, activation='relu')(x)

        s2 = tf.keras.layers.Dense(10, activation='relu')(x)

        s3 = tf.keras.layers.concatenate([s1, s2])

        s1 = tf.keras.layers.Dense(2, activation='linear')(s3)

        out = s1

        nn_improve_2 = tf.keras.Model(inputs=inp, outputs=out, name='improve_2')

        nn_improve_2.compile(optimizer='adam', loss='mae', metrics='mae')
        
        nn_improve_2.fit(self.X_train, self.y_train, batch_size=10, epochs=100, verbose=0)

        self.model = nn_improve_2

    def train_model_4(self):
        inp = tf.keras.layers.Input(shape=(99,), name='input')
        x = tf.keras.layers.Dense(200, activation='relu')(inp)
        x = tf.keras.layers.Dropout(.2, input_shape=(x.shape))(x)
        # x = tf.keras.layers.Dense(25, activation='relu')(x)
        # x = tf.keras.layers.Dropout(.2, input_shape=(x.shape))(x)

        # sp1 = tf.keras.layers.Lambda(lambda x: tf.slice(x, (-1,0), (-1, 150)))(x)
        # sp2 = tf.keras.layers.Lambda(lambda x: tf.slice(x, (-1,150), (-1, 150)))(x)

        s1 = tf.keras.layers.Dense(50, activation='relu')(x)
        s1 = tf.keras.layers.Dropout(.2, input_shape=(s1.shape))(s1)
        # s1 = tf.keras.layers.Dense(12, activation='relu')(x)
        # s1 = tf.keras.layers.Dropout(.2, input_shape=(s1.shape))(s1)
        # s1 = tf.keras.layers.Dense(6, activation='relu')(s1)

        s2 = tf.keras.layers.Dense(50, activation='relu')(x)
        s2 = tf.keras.layers.Dropout(.2, input_shape=(s2.shape))(s2)
        # s2 = tf.keras.layers.Dense(12, activation='relu')(x)
        # s2 = tf.keras.layers.Dropout(.2, input_shape=(s2.shape))(s2)
        # s2 = tf.keras.layers.Dense(6, activation='relu')(s2)

        out1 = tf.keras.layers.Dense(1, activation='relu', name='precision')(s1)
        out2 = tf.keras.layers.Dense(1, activation='relu', name='log_precision')(s2)

        # s3 = tf.keras.layers.concatenate([s1, s2])

        # s1 = tf.keras.layers.Dense(2, activation='relu')(s3)

        # out = s1

        nn_improve_2 = tf.keras.Model(inputs=inp, outputs=[out1, out2], name='improve_2')

        nn_improve_2.compile(loss='mae', metrics='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

        y_train_precision = self.y_train['Precision']
        y_train_log_precision = self.y_train['log_Precision']

        nn_improve_2.fit({"input": self.X_train},
                         {'precision': y_train_precision, 'log_precision': y_train_log_precision},
                         batch_size=10, epochs=100)

        self.model = nn_improve_2

    def get_mae_precision_branch_network(self):
        self.precision_mae = mean_absolute_error(np.asarray(self.y_test)[:, 0], self.y_pred[0])
        return self.precision_mae

    def get_mae_log_precision_branch_network(self):
        self.log_precision_mae = mean_absolute_error(np.asarray(self.y_test)[:, 1], self.y_pred[1])
        return self.log_precision_mae

    def get_model_plot(self, png_name='model'):
        tf.keras.utils.plot_model(self.model, f"{png_name}.png", show_shapes=True)

    def get_summary(self):
        return self.model.summary()
        
    def save_mode(self, model_name_location):
        self.model.save(model_name_location.split('.')[0])

