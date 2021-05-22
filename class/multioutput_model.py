import numpy as np
import pandas as pd
from model_class import Model
from sklearn.metrics import mean_absolute_error


class Multioutput(Model):
    def __init__(self, X_train, y_train, X_test, y_test, model_kwargs):
        super(Multioutput, self).__init__(X_train, y_train, X_test, y_test, model_kwargs)
        self.precision_mae = None
        self.log_precision_mae = None

    def get_predictions(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def get_mae_precision(self):
        self.precision_mae = mean_absolute_error(np.asarray(self.y_test)[:, 0], self.y_pred[:, 0])
        return self.precision_mae

    def get_mae_log_precision(self):
        self.log_precision_mae = mean_absolute_error(np.asarray(self.y_test)[:, 1], self.y_pred[:, 1])
        return self.log_precision_mae

    def get_mae_for_csv(self):
        return self.precision_mae, self.log_precision_mae

    def get_df_with_predictions_for_csv(self, algorithm_no, fold, labels=None):
        if labels is None:
            labels = ['Precision', 'log_Precision']

        y_pred_df = pd.DataFrame(self.y_pred)
        y_test_df = pd.DataFrame(self.y_test)

        y_pred_df.index = y_test_df.index
        y_pred_df.columns = [f'{label}_prediction' for label in labels]
        y_test_df.columns = [f'{label}_real' for label in labels]

        real_pred_df = y_pred_df.join(y_test_df)

        real_pred_df['Index_col'] = real_pred_df.index
        real_pred_df['Algorithm'] = algorithm_no
        real_pred_df['Fold'] = fold

        return real_pred_df


