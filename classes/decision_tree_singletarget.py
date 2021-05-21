from singleoutput_model import Singleoutput
from sklearn.tree import DecisionTreeRegressor


class DecisionTree_Single(Singleoutput):
    def __init__(self, X_train, y_train, X_test, y_test, model_kwargs, target):
        super(DecisionTree_Single, self).__init__(X_train, y_train, X_test, y_test, model_kwargs, target)

    def train_model(self):
        model = DecisionTreeRegressor(**self.model_kwargs).fit(self.X_train, self.y_train)
        self.model = model
