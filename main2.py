from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sys
import data_loading
import linear
import cnn
import rnn
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from param import Param
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Activation, Dropout, Input, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from keras.callbacks import EarlyStopping
#import output.plots as display
import tensorflow as tf
from keras import backend as K
from nn import NN
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout
from nn import NN
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit


def windowed_means(out_features, param):
    """
    Windowed means features extraction method
    :param out_features: epoched data in 3D shape (epochs_count x channels_count x values_count)
    :param param: configuration object
    :return: 2D vector with calculated features (epochs_count x features_count)
    """
    sampling_fq = param.t_max * 1000 + 1
    temp_wnd = np.linspace(param.min_latency, param.max_latency, param.steps + 1)
    intervals = np.zeros((param.steps, 2))
    for i in range(0, temp_wnd.shape[0] - 1):
        intervals[i, 0] = temp_wnd[i]
        intervals[i, 1] = temp_wnd[i + 1]
    intervals = intervals - param.t_min
    output_features = []
    for i in range(out_features.shape[0]):
        feature = []
        for j in range(out_features.shape[1]):
            time_course = out_features[i][j]
            for k in range(intervals.shape[0]):
                borders = intervals[k] * sampling_fq
                feature.append(np.average(time_course[int(borders[0] - 1):int(borders[1] - 1)]))
        output_features.append(feature)
    out = preprocessing.scale(np.array(output_features), axis=1)
    return out


def print_help():
    print("Usage: python main.py <classifier>\n")
    print("You can choose from these classifiers: lda, lor, svm, cnn, rnn, randfor, xgb\n")


"""
The program is executable from the command line using this file with one argument represents the choice of classifier. 
The command has the following form:

python main.py <classifier>

The user can choose from 7 types of classifiers, so the possible variants of the command are:

python main2.py lda
python main2.py svm
python main2.py lor
python main2.py cnn
python main2.py rnn
python main2.py randfor
python main2.py xgb

All other parameters are configurable in the param.py file.
"""

if len(sys.argv) != 2:
    print("The wrong number of command line arguments!\n")
    print_help()
    exit(1)

classifier = sys.argv[1]

if classifier != 'cnn' and classifier != 'rnn' and classifier != 'lda' and classifier != 'svm' and classifier != 'lor' and classifier != 'randfor' and classifier != 'xgb':
    print("The wrong choice of classifier!\n")
    print_help()
    exit(1)

param = Param()

X, Y = data_loading.read_data(param)

if classifier == 'cnn':
    X = np.expand_dims(X, 3)
elif classifier == 'lda' or classifier == 'svm' or classifier == 'lor' or classifier == 'randfor' or classifier == 'xgb':
    X = windowed_means(X, param)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=param.test_part,
                                                    random_state=0, shuffle=True)

val = round(param.validation_part * x_train.shape[0])

## monte-carlo split https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
# shuffle_split = ShuffleSplit(n_splits=param.cross_val_iter, test_size=val, random_state=0)

# Time-series split https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
shuffle_split = TimeSeriesSplit(n_splits=param.cross_val_iter, test_size=100, gap=0)

val_results = []
test_results = []
iter_counter = 0

# Monte-carlo cross-validation
for train, validation in shuffle_split.split(x_train):
    iter_counter = iter_counter + 1
    print(iter_counter, "/", param.cross_val_iter, " cross-validation iteration")

    if classifier == 'cnn':
        model = cnn.CNN(x_train.shape[1], x_train.shape[2], param)
    elif classifier == 'rnn':
        model = rnn.RNN(x_train.shape[1], x_train.shape[2], param)
    elif classifier == 'lor':
        model = linear.LinearClassifier(LogisticRegression())
    elif classifier == 'lda':
        model = linear.LinearClassifier(LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))
    elif classifier == 'randfor':
        model = linear.LinearClassifier(RandomForestClassifier(n_estimators=100, max_depth = 3))
    elif classifier == 'xgb':
        
    
    #     param_grid_old = dict(
    #     n_jobs=[16],
    #     learning_rate=[0.1, 0.5],
    #     max_depth=[5, 10, 15], 
    #     n_estimators=[100, 500, 1000],
    #     subsample=[0.2, 0.8, 1.0],
    #     gamma=[0.05, 0.5],
    #     scale_pos_weight=[0, 1],
    #     reg_alpha=[0, 0.5],
    #     reg_lambda=[1, 0],
    # )

    #     param_grid = dict(
    #     n_jobs=[16],
    #     max_depth=[5, 10], 
    #     n_estimators=[100, 200],
    #     gamma=[0.05, 0.5],
    #     reg_alpha=[0, 0.5],
    # )

    #     first_model = XGBClassifier(random_state=1, verbosity=1, use_label_encoder=False, eval_metric='mlogloss')
    #     grid_search = GridSearchCV(estimator=first_model,
    #                             param_grid=param_grid,
    #                             scoring='neg_root_mean_squared_error',
    #                             )

    #     model = grid_search.fit(x_train, y_train[:, 0])
    #     print('Optimum parameters', model.best_params_)

        model = linear.LinearClassifier(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
        n_estimators=200, subsample=0.9, gamma=0.05, max_depth=10, n_jobs=16, reg_alpha=0))
    else:
        """
        param_grid = {
        'cache_size' : [100, 200]
        }

        first_model = SVC()
        grid_search = GridSearchCV(estimator=first_model,
                                param_grid=param_grid
                                )

        model = grid_search.fit(x_train, y_train[:, 0])
        print('Optimum parameters', model.best_params_)
        """

        model = linear.LinearClassifier(SVC(cache_size=500))


    validation_metrics = model.fit(x_train[train], y_train[train], x_train[validation], y_train[validation])
    val_results.append(validation_metrics)

    test_metrics = model.evaluate(x_test, y_test)
    test_results.append(test_metrics)

print("\nClassifier: ", classifier)

avg_val_results = np.round(np.mean(val_results, axis=0) * 100, 2)
avg_val_results_std = np.round(np.std(val_results, axis=0) * 100, 2)

print("Averaged validation results with averaged std in brackets:")
print("AUC: ", avg_val_results[0], "(", avg_val_results_std[0], ")")
print("accuracy: ", avg_val_results[1], "(", avg_val_results_std[1], ")")
print("precision: ", avg_val_results[2], "(", avg_val_results_std[2], ")")
print("recall: ", avg_val_results[3], "(", avg_val_results_std[3], ")")

print("\n##############################\n")

avg_test_results = np.round(np.mean(test_results, axis=0) * 100, 2)
avg_test_results_std = np.round(np.std(test_results, axis=0) * 100, 2)

print("Averaged test results with averaged std in brackets: ")
print("AUC: ", avg_test_results[0], "(", avg_test_results_std[0], ")")
print("accuracy: ", avg_test_results[1], "(", avg_test_results_std[1], ")")
print("precision: ", avg_test_results[2], "(", avg_test_results_std[2], ")")
print("recall: ", avg_test_results[3], "(", avg_test_results_std[3], ")")
