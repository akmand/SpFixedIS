import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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
        df = shuffle(df, n_samples=n_samples, random_state=random_state)

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





def SpFixedIS_by_instance_run(dataset_name, instance_range, model_catalog,
                              data_directory,
                              result_directory,
                              save_file_suffix = '',
                              scoring=accuracy_score,
                              cv_repeats = 10, cv_folds =5, random_state=999):

    '''
    This function performs fixed instance selection using SpFixedIS using a list of specified number of instances and
    returns the results in a csv.

    :param dataset_name: string, name of the dataset
    :param instance_range: list of integers, the list of the number of instances to test
    :param model_catalog: dictionary, key is used to reference the model, the value is a sklearn model.
    Ensure the random state is set for the model where applicable before parsing it into the function
    :param data_directory: string, the path the the data directory
    :param result_directory: string, the path the results directory
    :param save_file_suffix: string, optional, default is empty, append text to the file name.
    :param scoring: sklearn metric, default is accuracy_score
    :param cv_repeats: integer, default is 10, number of repetitions for cross validation
    :param cv_folds: integer, default is 5, number of cross validation folds
    :param random_state: integer, random state to use.
    :return: None. File will be saved to specified results directory
    '''

    results_dataset = pd.DataFrame(columns=['dataset', 'rep_no', 'fold_no', 'model', 'method',
                                            'num_ins', 'accuracy', 'run_time', 'train_index', 'test_index',
                                            'selected_index'])

    x, y = prepare_dataset_for_modeling(dataset_name, pred_type='c',  data_directory=data_directory,n_samples_max=50000)

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

            for model_name, model in model_catalog.items():
                # evaluate the full training set
                full_start_time = time.time()
                model.fit(X_train, y_train)
                full_accuracy = scoring(y_test, model.predict(X_test))
                full_time = round((time.time() - full_start_time) / 60, 2)

                results_dataset = results_dataset.append(
                    {'dataset': dataset_name, 'rep_no': rep_no, 'fold_no': fold_no, 'model': model_name,
                     'method': 'full', 'num_ins': 0, 'accuracy': full_accuracy, 'run_time': full_time,
                     'train_index': train_index, 'test_index': test_index, 'selected_index': train_index},
                    ignore_index=True)

                print(results_dataset[['dataset', 'rep_no', 'fold_no', 'model', 'method',
                                              'num_ins', 'accuracy', 'run_time']].tail(1))

                for instance_number in instance_range:

                    # evaluate random stratified selection
                    random_time_start = time.time()
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=instance_number,
                                                 test_size=len(X_train)-instance_number,
                                                 random_state=random_state + rep_no + instance_number)

                    for random_index, unused_index in sss.split(X_train, y_train):
                        random_index = random_index

                    model.fit(X_train[random_index, :], y_train[random_index])
                    random_accuracy = scoring(y_test, model.predict(X_test))
                    random_time = round((time.time() - random_time_start) / 60, 2)

                    results_dataset = results_dataset.append(
                        {'dataset': dataset_name, 'rep_no': rep_no, 'fold_no': fold_no, 'model': model_name,
                         'method': 'random', 'num_ins': instance_number, 'accuracy': random_accuracy, 'run_time': random_time,
                         'train_index': train_index, 'test_index': test_index, 'selected_index': random_index},
                        ignore_index=True)
                    print(results_dataset[['dataset', 'rep_no', 'fold_no', 'model', 'method',
                                           'num_ins', 'accuracy', 'run_time']].tail(1))
                    # evaluate spsa selection
                    sp_engine = SpFixedIS.SpFixedIS(X_train, y_train, scoring=scoring, wrapper=model)
                    sp_run = sp_engine.run(num_instances=instance_number, print_freq=50)
                    sp_results = sp_run.results

                    selected_index = sp_results.get('instances')
                    IS_time = sp_results.get('run_time')
                    model.fit(X_train[selected_index, :], y_train[selected_index])
                    IS_accuracy = scoring(y_test, model.predict(X_test))

                    results_dataset = results_dataset.append(
                        {'dataset': dataset_name, 'rep_no': rep_no, 'fold_no': fold_no, 'model': model_name,
                         'method': 'spsa', 'num_ins': instance_number, 'accuracy': IS_accuracy, 'run_time': IS_time,
                         'train_index': train_index, 'test_index': test_index, 'selected_index': selected_index},
                        ignore_index=True)

                    print(results_dataset[['dataset', 'rep_no', 'fold_no', 'model', 'method',
                                           'num_ins', 'accuracy', 'run_time']].tail(1))

                results_dataset.to_csv(result_directory + dataset_name+ save_file_suffix +'.csv', index=False)


