import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import time
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import SpFixedIS
import os
import io
import requests
import ssl
from sklearn import preprocessing
from sklearn.utils import shuffle


def prepare_dataset_for_modeling(dataset_name,
                                 pred_type,
                                 data_directory=None,
                                 na_values='?',
                                 n_samples_max=None,
                                 random_state=999,
                                 drop_const_columns=True,
                                 scale_data=True):
    """
    ASSUMPTION 1: The target variable is the LAST column in the dataset.
    ASSUMPTION 2: First row in the file is the header row.
    :param dataset_name: name of the dataset - will be passed to pd.read_csv()
    :param pred_type: if 'c' for classification, y is assumed categorical and it will be label-encoded for model fitting
                      if 'r' for regression, y is assumed numerical
    :param data_directory: directory of the dataset. If None, the dataset will be read in from GitHub
    :param na_values: Additional strings to recognize as NA/NaN - will be passed to pd.read_csv()
    :param n_samples_max: max no. of instances to sample (if not None)
    :param random_state: seed for shuffling (and sampling) instances
    :param drop_const_columns: if True, drop constant-value columns (*after* any sampling)
    :param scale_data: whether the descriptive features (and y also if regression) are to be min-max scaled
    :return: x and y NumPy arrays ready for model fitting
    """

    if pred_type not in ['c', 'r']:
        raise ValueError("Error: prediction type needs to be either 'c' for classification or 'r' for regression.")

    if data_directory:
        # read in from local directory
        df = pd.read_csv(data_directory + dataset_name + '.csv', na_values=na_values, header=0)
    else:
        # read in the data file from GitHub into a Pandas data frame
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context
        github_location = 'https://raw.githubusercontent.com/vaksakalli/datasets/master/'
        dataset_url = github_location + dataset_name.lower()
        df = pd.read_csv(io.StringIO(requests.get(dataset_url).content.decode('utf-8')), na_values=na_values, header=0)

    # drop missing values before (any) sampling
    df = df.dropna()

    # shuffle dataset in case of a pattern and also subsample if requested
    # but do not sample more than the available number of observations (*after* dropping missing values)
    # n_samples_max = None results in no sampling; just shuffling
    n_observations = df.shape[0]  # no. of observations in the dataset
    n_samples = n_observations  # initialization - no. of observations after (any) sampling
    if n_samples_max and (n_samples_max < n_observations):
        # do not sample more rows than what is in the dataset
        n_samples = n_samples_max
    #df = shuffle(df, n_samples=n_samples, random_state=random_state)

    if drop_const_columns:
        # drop constant columns (after sampling)
        df = df.loc[:, df.nunique() > 1]

    # last column is y (target feature)
    y = df.iloc[:, -1].values
    # everything else is x (set of descriptive features)
    x = df.iloc[:, :-1]

    # get all columns that are objects
    # these are assumed to be nominal categorical
    categorical_cols = x.columns[x.dtypes == object].tolist()

    # if a categorical feature has only 2 levels:
    # encode it as a single binary variable
    for col in categorical_cols:
        n = len(x[col].unique())
        if n == 2:
            x[col] = pd.get_dummies(x[col], drop_first=True)

    # for categorical features with >2 levels: use one-hot-encoding
    # below, numerical columns will be untouched
    x = pd.get_dummies(x).values

    if scale_data:
        # scale x between 0 and 1
        x = preprocessing.MinMaxScaler().fit_transform(x)
        if pred_type == 'r':
            # also scale y between 0 and 1 for regression problems
            y = preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

    if pred_type == 'c':
        # label-encode y for classification problems
        y = preprocessing.LabelEncoder().fit_transform(y)

    return x, y


def centroid(x, y):
    '''
    This function returns the index of the instances closest to the centroid of each class label

    :param x: 2d numpy array, a numpy array for the descriptive features of a dataset
    :param y: 1d numpy array, a numpy array for the target features of a dataset
    :return 1d numpy array, a numpy array containing the indices of the selected instances
    '''

    centroids = np.array([], dtype='int64')
    for classes in np.unique(y):
        nbrs = NearestNeighbors(n_neighbors=1).fit(x[y == classes])
        temp = x[y == classes].mean(axis=0)

        nearest = nbrs.kneighbors([temp, temp])[1][0][0]

        centroids = np.append(centroids, int(np.where(np.all(x == x[y == classes][nearest], axis=1))[0][0]))

    return np.array(centroids, dtype='int64')

def dm(a, b):
    '''
    This function calculates the euclidean distance between 2 observations

    :param a: 1d numpy array
    :param b: 1d numpy array
    :return integer
    '''

    return np.linalg.norm(a - b)


def FCNN1(x, y):
    '''
    This function performs Fast Condensed Nearest Neighbours 1.

    :param x: 2d numpy array, a numpy array for the descriptive features of a dataset
    :param y: 1d numpy array, a numpy array for the target features of a dataset
    :return 1d numpy array, a numpy array containing the indices of the selected instances

    '''

    S = np.array([], dtype="int64")

    nearest = {}
    for i in range(len(x)):
        nearest[i] = -1

    delta_S = centroid(x, y)

    cc = 0
    while len(delta_S) > 0:
        cc = cc + 1
        S = np.append(S, delta_S)

        rep = {}
        for i in S:
            rep[i] = -1

        for q in np.setdiff1d(np.arange(len(x)), S):

            # calculating the nearest neighbor to each element in S within the remaining set
            for p in delta_S:
                if nearest[q] == -1:
                    nearest[q] = p
                    # print(cc)
                else:
                    if dm(x[nearest[q]], x[q]) > dm(x[p], x[q]):
                        nearest[q] = p

            # check to see whether the nearest neighbor is an enemy and the closest point in T to the representative in S
            if y[nearest[q]] != y[q]:
                if rep[nearest[q]] == -1:
                    rep[nearest[q]] = q

                else:
                    if dm(x[nearest[q]], x[q]) < dm(x[nearest[q]], x[rep[nearest[q]]]):
                        rep[nearest[q]] = q

        delta_S = np.array([], dtype="int64")

        for p in S:
            if rep[p] != -1:
                delta_S = np.append(delta_S, rep[p])
    return S


def SpFixedIS_comp_run(dataset_name,
                       data_directory,
                       result_directory,
                       model = KNeighborsClassifier(n_neighbors=1),
                       scoring=accuracy_score,
                       cv_repeats = 10, cv_folds =5, random_state=999):

    '''
    This function performs FCNN1 on a dataset, storing the index of selected instances, before matching the number of
    selected instances for SPFixedIS on the same corresponding cross validation splits. The results stored are the
    number of instances selected, the index of the selected instances for both methods as well as the test train index.

    :param dataset_name: string, name of the dataset
    :param data_directory: string, the path to the data directory
    :param result_directory: string, the path to the results directory
    :param model: sklearn classification model, default is KNeighborsClassifier(n_neighbors=1)
    :param scoring: sklearn metric, default is accuracy_score
    :param cv_repeats: integer, number of cross validation repetitions, default is 10
    :param cv_folds: integer, number of cross validation folds, default is 5
    :param random_state: integer, random seed to use, default is 999
    :return: None. File will be saved in specified directory
    '''

    results_dataset = pd.DataFrame(columns=['dataset', 'rep_no', 'fold_no', 'method',
                                            'num_ins', 'accuracy', 'run_time', 'train_index', 'test_index',
                                            'selected_index'])

    x, y = prepare_dataset_for_modeling(dataset_name, pred_type='c', data_directory=data_directory)

    # loop over number of repeats
    for rep_no in range(cv_repeats):
        sk_fold = StratifiedKFold(n_splits=cv_folds,
                                  shuffle=True,
                                  random_state=random_state + rep_no)

        # loop over the folds
        fold_no = 0
        for train_index, test_index in sk_fold.split(x, y):
            fold_no += 1

            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # evaluate the full training set
            full_start_time = time.time()
            model.fit(X_train, y_train)
            full_accuracy = scoring(y_test, model.predict(X_test))
            full_time = round((time.time() - full_start_time) / 60, 2)

            results_dataset = results_dataset.append(
                {'dataset': dataset_name, 'rep_no': rep_no, 'fold_no': fold_no,
                 'method': 'full', 'num_ins': 0, 'accuracy': full_accuracy, 'run_time': full_time,
                 'train_index': train_index, 'test_index': test_index, 'selected_index': train_index},
                ignore_index=True)

            print(results_dataset[['dataset', 'rep_no', 'fold_no', 'method',
                                   'num_ins', 'accuracy', 'run_time']].tail(1))

            # run FCNN1
            fcnn_time_start = time.time()
            fcnn_index = FCNN1(X_train, y_train)
            model.fit(X_train[fcnn_index, :], y_train[fcnn_index])
            fcnn_scoring = scoring(y_test, model.predict(X_test))
            fcnn_time= round((time.time() - fcnn_time_start) / 60, 2)

            fcnn_num_selected = len(fcnn_index)

            results_dataset = results_dataset.append(
                {'dataset': dataset_name, 'rep_no': rep_no, 'fold_no': fold_no,
                 'method': 'FCNN1', 'num_ins': fcnn_num_selected, 'accuracy': fcnn_scoring, 'run_time': fcnn_time,
                 'train_index': train_index, 'test_index': test_index, 'selected_index': fcnn_index},
                ignore_index=True)

            print(results_dataset[['dataset', 'rep_no', 'fold_no', 'method',
                                   'num_ins', 'accuracy', 'run_time']].tail(1))

            # evaluate spsa selection
            sp_engine = SpFixedIS.SpFixedIS(X_train, y_train, scoring=scoring, wrapper=model)
            sp_run = sp_engine.run(num_instances=fcnn_num_selected)
            sp_results = sp_run.results

            selected_index = sp_results.get('instances')
            IS_time = sp_results.get('run_time')
            model.fit(X_train[selected_index, :], y_train[selected_index])
            IS_accuracy = scoring(y_test, model.predict(X_test))

            results_dataset = results_dataset.append(
                {'dataset': dataset_name, 'rep_no': rep_no, 'fold_no': fold_no,
                 'method': 'spsa', 'num_ins': fcnn_num_selected, 'accuracy': IS_accuracy, 'run_time': IS_time,
                 'train_index': train_index, 'test_index': test_index, 'selected_index': selected_index},
                ignore_index=True)

            print(results_dataset[['dataset', 'rep_no', 'fold_no', 'method',
                                   'num_ins', 'accuracy', 'run_time']].tail(1))

            results_dataset.to_csv(result_directory + dataset_name+ '_comp.csv', index=False)

