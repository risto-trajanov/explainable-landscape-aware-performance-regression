import abc


class Model:
    def __init__(self, X_train, y_train, X_test, y_test, model_kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_kwargs = model_kwargs
        self.model = None
        self.y_pred = None


    @abc.abstractmethod
    def train_model(self):
        pass

    def get_model(self):
        return self.model

    @abc.abstractmethod
    def get_predictions(self):
        """
        :return: Return the predictions for X_test
        """
        pass

    @abc.abstractmethod
    def get_mae_for_csv(self):
        """

        :return: Return Mean Absolute Error in format precision_mae, log_precision_mae
        """
        pass

    @abc.abstractmethod
    def get_df_with_predictions_for_csv(self, algorithm_no, fold):
        """

        :return: Return Data Frame with Index from y_test, Precision_prediction	log_Precision_prediction	Precision_real
                                                        log_Precision_real	Index_col	Algorith	Fold
        """
        pass
