from singleoutput_model import Singleoutput
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressor_Single(Singleoutput):
    def __init__(self, X_train, y_train, X_test, y_test, model_kwargs, target):
        super(RandomForestRegressor_Single, self).__init__(X_train, y_train, X_test, y_test, model_kwargs, target)

    def train_model(self):
        model = RandomForestRegressor(**self.model_kwargs).fit(self.X_train, self.y_train)
        self.model = model
