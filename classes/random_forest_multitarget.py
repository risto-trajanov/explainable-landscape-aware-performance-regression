from multioutput_model import Multioutput
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressor_Multi(Multioutput):
    def __init__(self, X_train, y_train, X_test, y_test, model_kwargs):
        super(RandomForestRegressor_Multi, self).__init__(X_train, y_train, X_test, y_test, model_kwargs)

    def train_model(self):
        model = RandomForestRegressor(**self.model_kwargs).fit(self.X_train, self.y_train)
        self.model = model
