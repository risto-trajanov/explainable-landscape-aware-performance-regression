from multioutput_model import Multioutput
from sklearn.tree import DecisionTreeRegressor


class DecisionTree_Multi(Multioutput):
    def __init__(self, X_train, y_train, X_test, y_test, model_kwargs):
        super(DecisionTree_Multi, self).__init__(X_train, y_train, X_test, y_test, model_kwargs)

    def train_model(self):
        model = DecisionTreeRegressor(**self.model_kwargs).fit(self.X_train, self.y_train)
        self.model = model
