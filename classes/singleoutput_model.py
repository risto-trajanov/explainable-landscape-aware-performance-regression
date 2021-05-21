import numpy as np
import pandas as pd
from model_class import Model
from sklearn.metrics import mean_absolute_error


class Singleoutput(Model):
    def __init__(self, X_train, y_train, X_test, y_test, model_kwargs, target):
        super(Singleoutput, self).__init__(X_train, y_train, X_test, y_test, model_kwargs)
        self.mae = None
        self.target = target

    def get_predictions(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def get_mae(self):
        predictions = [value for value in self.y_pred]
        self.mae = mean_absolute_error(np.asarray(self.y_test), np.asarray(self.y_pred))
        return self.mae

    def get_mae_for_csv(self):
        return self.mae

    def get_df_with_predictions_for_csv(self, algorithm_no, fold):
        y_pred_df = pd.DataFrame(self.y_pred)
        y_test_df = pd.DataFrame(self.y_test)

        y_pred_df.index = y_test_df.index

        y_pred_df.columns = [self.target + '_prediction']
        y_test_df.columns = [self.target + '_real']

        real_pred_df = y_pred_df.join(y_test_df)

        real_pred_df['Index_col'] = real_pred_df.index
        real_pred_df['Algorithm'] = algorithm_no
        real_pred_df['Fold'] = fold

        return real_pred_df


