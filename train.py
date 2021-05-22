import os
import sys
import pickle
import pandas as pd
import numpy as np
import sys
from sklearn.feature_selection import chi2, SelectKBest, f_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import settings as project_settings

target_data_folder = project_settings.target_data_folder
features_data_folder = project_settings.features_data_folder
class_folder = project_settings.class_folder
result_folder = project_settings.result_folder
algorithm_len = project_settings.algorithm_len

sys.path.append(class_folder)

from decision_tree_singletarget import DecisionTree_Single
from decision_tree_multitarget import DecisionTree_Multi
from random_forest_singletarget import RandomForestRegressor_Single
from random_forest_multitarget import RandomForestRegressor_Multi
from dnn_model import DNN
from dnn_single_model import DNN_Single

labels =pd.read_csv(f"{target_data_folder}labels.txt",sep=';',index_col=False)
sys_min = sys.float_info.min

def create_data(df,df_perf,labels):
    df2 = df_perf.assign(label = labels['x'])
    df2 = df2.rename(columns={'1' : 'Precision'})
    data = df.join(df2.set_index('label'))

    return data

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[df.replace([-np.inf], sys.float_info.min).notnull().all(axis=1)]

    return df[indices_to_keep].astype(np.float32)

def get_data_for_algorith(algorithm, no_fold):
    df_perf = pd.read_csv(f"{target_data_folder}performance_0_I{algorithm}.txt",sep='\t')
    df_train = pd.read_csv(f"{features_data_folder}train_{no_fold}_fused.csv",sep='\t', index_col=0)
    df_test = pd.read_csv(f"{features_data_folder}test_{no_fold}_fused.csv",sep='\t', index_col=0)
    df_perf = df_perf.iloc[:,2:3]

    rez_test = create_data(df_test,df_perf,labels)
    rez_train = create_data(df_train,df_perf,labels)

    return rez_train, rez_test

# Check valid data
def valid_data(df):
    if len(df[df.isin([np.nan, np.inf, -np.inf]).any(1)]) == 0:
        return True
    else:
        return False

def add_log_performance(df):
    df['log_Precision'] = np.log10(df.iloc[:, -1] + 1)
    return df

def get_features(X_train, y_train):
    selected_features = []
    for i in range(0, len(y_train.columns)):
        selector = SelectKBest(f_regression, k=10)
        selector.fit(X_train, y_train.iloc[:,i])
        #selected_features.append(list(selector.scores_))
        cols = selector.get_support(indices=True)
        selected_features.append(cols)

    features = set()
    for array in selected_features:
        for feature in array:
            features.add(feature)
    return list(features)

def get_data(algorithm_no, fold):

    if not isinstance(algorithm_no, list) and not isinstance(algorithm_no, tuple):
        train_df, test_df = get_data_for_algorith(algorithm_no, fold)

        train_df_with_log = add_log_performance(train_df)
        test_df_with_log = add_log_performance(test_df)

        train_df_with_log_clean = clean_dataset(train_df_with_log)
        test_df_with_log_clean = clean_dataset(test_df_with_log)

        if valid_data(train_df_with_log_clean) and valid_data(test_df_with_log_clean):
            X_train = train_df_with_log_clean.iloc[:, :-2]
            y_train_labels = train_df_with_log_clean.iloc[:, -2:]

            X_test = test_df_with_log_clean.iloc[:, :-2]
            y_test_labels = test_df_with_log_clean.iloc[:, -2:]

            return X_train, y_train_labels, X_test, y_test_labels
        else:
            raise Exception("Invalid Data")

    else:
        data_train, data_test = [], []

        for alg in algorithm_no:
            d_train, d_test = get_data_for_algorith(alg, fold)
            data_train.append(d_train)
            data_test.append(d_test)

        merged_good_train = data_train[0]
        merged_good_test = data_test[0]

        for i in range(1, len(algorithm_no)):
            merged_train = pd.merge(merged_good_train, data_train[i], how='inner', left_index=True, right_index=True,
                                    suffixes=(f'_x{i}', f'_y{i}'))
            merged_test = pd.merge(merged_good_test, data_test[i], how='inner', left_index=True, right_index=True,
                                   suffixes=(f'_x{i}', f'_y{i}'))

            merged_good_train = merged_train[list(merged_train.columns[0:(100+i-1)]) + [merged_train.columns[-1]]]
            merged_good_test = merged_test[list(merged_test.columns[0:(100+i-1)]) + [merged_test.columns[-1]]]

        # Fix the column names
        index = pd.Index(list(data_train[0].columns[:99]) + [f'Precision_alg{alg}' for alg in algorithm_no])
        merged_good_train.columns = index
        merged_good_test.columns = index

        X_train = merged_good_train.iloc[:, :-len(algorithm_no)]
        y_train_labels = merged_good_train.iloc[:, -len(algorithm_no):]

        X_test = merged_good_test.iloc[:, :-len(algorithm_no)]
        y_test_labels = merged_good_test.iloc[:, -len(algorithm_no):]


def get_model(X_train, y_train, X_test, y_test, model_name, model_kwargs=None, single_output=True, target=None):
    if model_kwargs is None:
        model_kwargs = {}
    if single_output:
        if model_name == 'Xgboost':
            return Xgboost_Single(X_train, y_train, X_test, y_test, model_kwargs, target)
        elif model_name == 'nn':
            return DNN_Single(X_train, y_train, X_test, y_test, target)
        elif model_name == 'decision_tree':
            return DecisionTree_Single(X_train, y_train, X_test, y_test, model_kwargs, target)
        elif model_name == 'linear_regression':
            return LinearRegression_Single(X_train, y_train, X_test, y_test, model_kwargs, target)
        elif model_name == 'k_neighbors':
            return KNeighborsRegressor_Single(X_train, y_train, X_test, y_test, model_kwargs, target)
        elif model_name == 'random_forest':
            return RandomForestRegressor_Single(X_train, y_train, X_test, y_test, model_kwargs, target)
        else:
            pass
    else:
        if model_name == 'Xgboost':
            return Xgboost_Multi(X_train, y_train, X_test, model_kwargs, y_test)
        elif model_name == 'nn':
            return DNN(X_train, y_train, X_test, y_test)
        elif model_name == 'decision_tree':
            return DecisionTree_Multi(X_train, y_train, X_test, y_test, model_kwargs)
        elif model_name == 'linear_regression':
            return LinearRegression_Multi(X_train, y_train, X_test, y_test, model_kwargs)
        elif model_name == 'k_neighbors':
            return KNeighborsRegressor_Multi(X_train, y_train, X_test, y_test, model_kwargs)
        elif model_name == 'random_forest':
            return RandomForestRegressor_Multi(X_train, y_train, X_test, y_test, model_kwargs)
        else:
            pass

def dimension_reduction(x, n, dim_reduction_alg):
    alg = dim_reduction_alg.lower()
    alg_dict = {
        'pca': PCA,
        'svd': TruncatedSVD,
        'isomap': Isomap,
        'lle': LocallyLinearEmbedding
    }
    model = alg_dict[alg]
    return model(n_components=n).fit_transform(x)

def run_50_folds_on_algorithm_multi_target(algorithm_no, feature_selection = False, model_name='Xgboost', model_kwargs=None, dim_reduction_n=None, dim_reduction_alg='None', fold_range=None):
    model_name_folder = model_name
    if dim_reduction_alg and dim_reduction_n:
        model_name_folder += f'_{dim_reduction_alg}_n_{dim_reduction_n}'
    if model_kwargs:
        model_name_folder += '_' + '_'.join(f'{k}={v}'for k, v in model_kwargs.items())
        model_kwargs_str = '_'.join(f'{k}={v}'for k, v in model_kwargs.items())

    if not isinstance(algorithm_no, list) and not isinstance(algorithm_no, tuple):
        labels = ['Precision', 'log_Precision']
        algorithm_name = str(algorithm_no)
    else:
        labels = [f'Precision_alg{i}' for i in algorithm_no]
        algorithm_name = '_'.join(map(str, algorithm_no))


    predictions_folder = f"{result_folder}predictions/predictions_alg_no_{algorithm_name}/multi_target_output/{model_name_folder}/"
    mae_folder = f"{result_folder}mae/multi_target_output/{model_name_folder}/"
    models_folder = f'{result_folder}models/multi_target_output/{model_name_folder}/'
    os.makedirs(predictions_folder, exist_ok=True)
    os.makedirs(mae_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    output_mae = []

    if fold_range is None:
        fold_range = range(0, 50)

    for i in fold_range:

        # try:
        X_train, y_train, X_test, y_test = get_data(algorithm_no, i)

        if feature_selection:

            features = get_features(X_train, y_train)

            X_train = X_train.iloc[:, features]
            X_test = X_test.iloc[:, features]

        if dim_reduction_alg and dim_reduction_n:
            X_train = dimension_reduction(X_train, dim_reduction_n, dim_reduction_alg)
            X_test = dimension_reduction(X_test, dim_reduction_n, dim_reduction_alg)

        model = get_model(X_train, y_train, X_test, y_test, model_name, model_kwargs, False, None)

        model.train_model()

        print(f"PRINTING RESULTS FOR: fold number{i+1} and algorithm {algorithm_name}\n")
        print("Testing score: \n")

        y_pred = model.get_predictions()

        precision_mae = model.get_mae_precision()
        print(f"Precision MAE: {precision_mae:.4f}")

        log_precision_mae = model.get_mae_log_precision()
        print(f"Log Precision MAE: {log_precision_mae:.4f}")

        fold_algorithm_mae = [i, algorithm_name, precision_mae, log_precision_mae]
        output_mae.append(fold_algorithm_mae)

        real_pred_df = model.get_df_with_predictions_for_csv(algorithm_name, i, labels)

        real_pred_df.to_csv(f"{predictions_folder}predictions_fold_no_{i}_alg_no_{algorithm_name}.csv")
        model_name_location = f'{models_folder}model_fold_no_{i}_alg_no_{algorithm_name}.pkl'

        if model_name == 'nn':
            model_name_location = f'{models_folder}model_fold_no_{i}_alg_no_{algorithm_name}'
            model.save_model(model_name_location)

        else:
            with open(model_name_location, 'wb') as file:
                pickle.dump(model, file)

    output_mae_df = pd.DataFrame(data=output_mae, columns=['Fold', 'Algorithm', 'Precision_mae', 'log_Precision_mae'])
    output_mae_df.to_csv(f'{mae_folder}mae_alg_no_{algorithm_name}_multi_output_model.csv')


def run_50_folds_on_algorithm_single_target(algorithm_no, feature_selection = False, model_name = 'Xgboost', model_kwargs=None, dim_reduction_n=None, dim_reduction_alg='None', fold_range=None):
    model_name_folder = model_name
    if dim_reduction_alg and dim_reduction_n:
        model_name_folder += f'_{dim_reduction_alg}_n_{dim_reduction_n}'
    if model_kwargs:
        model_name_folder += '_' + '_'.join(f'{k}={v}'for k, v in model_kwargs.items())

    if not isinstance(algorithm_no, list) and not isinstance(algorithm_no, tuple):
        labels = ['Precision', 'log_Precision']
        algorithm_name = str(algorithm_no)
    else:
        labels = [f'Precision_alg{i}' for i in algorithm_no]
        algorithm_name = '_'.join(map(str, algorithm_no))

    predictions_folder = f"{result_folder}predictions/predictions_alg_no_{algorithm_name}/single_output_models/{model_name_folder}/"
    mae_folder = f"{result_folder}mae/single_output_models/{model_name_folder}/"
    models_folder = f'{result_folder}models/single_output_models/{model_name_folder}/'
    os.makedirs(predictions_folder, exist_ok=True)
    os.makedirs(mae_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    output_mae = []

    if fold_range is None:
        fold_range = range(0, 50)

    for i in fold_range:
        #try:
        X_train, y_train_labels, X_test, y_test_labels = get_data(algorithm_no, i)

        if feature_selection:

            features = get_features(X_train, y_train,)

            X_train = X_train.iloc[:, features]
            X_test = X_test.iloc[:, features]

        if dim_reduction_alg and dim_reduction_n:
            X_train = dimension_reduction(X_train, dim_reduction_n, dim_reduction_alg)
            X_test = dimension_reduction(X_test, dim_reduction_n, dim_reduction_alg)

        for label in labels:

            y_train = y_train_labels[label]
            y_test = y_test_labels[label]

            model = get_model(X_train, y_train, X_test, y_test, model_name, model_kwargs, True, label)

            model.train_model()

            print(f"PRINTING RESULTS FOR: fold number{i+1} and algorithm {algorithm_name}\n")
            print("Testing score: \n")

            y_pred = model.get_predictions()

            mae = model.get_mae()
            print(f"{label} MAE: {mae:.4f}")

            fold_algorithm_mae = [i, algorithm_name, label, model_name, mae]
            output_mae.append(fold_algorithm_mae)

            real_pred_df = model.get_df_with_predictions_for_csv(algorithm_name, i)

            real_pred_df.to_csv(f"{predictions_folder}predictions_fold_no_{i}_alg_no_{algorithm_name}_label_{label}.csv")

            model_name_location = f'{models_folder}model_fold_no_{i}_alg_no_{algorithm_name}_label_{label}.pkl'
            if model_name == 'nn':
                model_name_location = f'{models_folder}model_fold_no_{i}_alg_no_{algorithm_name}_label_{label}'
                model.save_model(model_name_location)

            else:
                with open(model_name_location, 'wb') as file:
                    pickle.dump(model, file)

    output_mae_df = pd.DataFrame(data=output_mae, columns=['Fold', 'Algorithm', 'Label', 'Model', 'MAE'])
    output_mae_df.to_csv(f'{mae_folder}mae_alg_no_{algorithm_name}_single_output_models.csv')

# Decision Tree
for alg in range(0, algorithm_len):
    decision_tree_args = {'max_depth': 9, 'criterion': 'mae'}
    run_50_folds_on_algorithm_single_target(alg, model_name='decision_tree', model_kwargs=decision_tree_args)

    decision_tree_args = {'max_depth': 10, 'criterion': 'mae'}
    run_50_folds_on_algorithm_multi_target(alg, model_name='decision_tree', model_kwargs=decision_tree_args)

# Decision Tree Strong
# for alg in range(0, algorithm_len):
#     decision_tree_args = {'max_depth': 25, 'criterion': 'mae'}
#     run_50_folds_on_algorithm_single_target(alg, model_name='decision_tree', model_kwargs=decision_tree_args)
#
#     decision_tree_args = {'max_depth': 50, 'criterion': 'mae'}
#     run_50_folds_on_algorithm_multi_target(alg, model_name='decision_tree', model_kwargs=decision_tree_args)

# Random Forest
for alg in range(0, algorithm_len):
    random_forest_args = {'n_estimators': 10, 'max_depth': 7, 'criterion': 'mae'}
    run_50_folds_on_algorithm_single_target(alg, model_name='random_forest', model_kwargs=random_forest_args)

    random_forest_args = {'n_estimators': 20, 'max_depth': 7, 'criterion': 'mae'}
    run_50_folds_on_algorithm_multi_target(alg, model_name='random_forest', model_kwargs=random_forest_args)


# Random Forest Strong
# for alg in range(0, algorithm_len):
    # random_forest_args = {'n_estimators': 100, 'max_depth': 25, 'criterion': 'mae'}
    # run_50_folds_on_algorithm_single_target(alg, model_name='random_forest', model_kwargs=random_forest_args)
    #
    # random_forest_args = {'n_estimators': 200, 'max_depth': 25, 'criterion': 'mae'}
    # run_50_folds_on_algorithm_multi_target(alg, model_name='random_forest', model_kwargs=random_forest_args)

# Neural Network
for alg in range(0, algorithm_len):
    run_50_folds_on_algorithm_single_target(alg, model_name='nn')
    run_50_folds_on_algorithm_multi_target(alg, model_name='nn')

random_forest_args = {'n_estimators': 25, 'max_depth': 25, 'criterion': 'mae'}
run_50_folds_on_algorithm_single_target(range(0, algorithm_len), model_name='random_forest', model_kwargs=random_forest_args)

random_forest_args = {'n_estimators': 75, 'max_depth': 25, 'criterion': 'mae'}
run_50_folds_on_algorithm_multi_target(range(0, algorithm_len), model_name='random_forest', model_kwargs=random_forest_args)
