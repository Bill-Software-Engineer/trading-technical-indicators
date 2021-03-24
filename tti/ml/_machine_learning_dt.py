"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_dt.py
    Implements a Decision Tree classification model for the Machine Learning
    features of the tti library.
"""

import time
import datetime

from ._machine_learning_api import MachineLearningAPI
from ._machine_learning_data import MachineLearningData
from ..utils.constants import ALL_TI_FEATURES

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


class MachineLearningDT(MachineLearningAPI):
    """
    Machine Learning DT class implementation.
    """

    def __init__(self):

        super().__init__(model_type='DT')

    def mlTrainModel(self, input_data, pool_size=None, verbose=False):
        """
        Trains a DT machine learning model for prices direction predictions.

        Args:
            input_data (pandas.DataFrame): The input data. Required input
                columns are ``high``, ``low``, ``open``, ``close`` and
                ``volume``. The index is of type ``pandas.DatetimeIndex``. The
                minimum number of data required is 60 periods.

            pool_size (int, default=None): Pool size for parallel computing
                when concurrency applies.  When None, then the created
                processes are equal to the number of the available cpu cores.

            verbose (bool, default=False): If set to True, processing
                information is sent to the console.

        Raises:
            WrongTypeForInputParameter: Input argument has wrong type
            WrongValueForInputParameter: Unsupported value for input argument
            NotEnoughInputData: Not enough data for calculating the indicator
            TypeError: Type error occurred when validating the ``input_data``
            ValueError: Value error occurred when validating the ``input_data``
            NoFeaturesSelectedForMLData: No features selected for ML data
            NotEnoughDataForMachineLearningTraining: Not enough data for ML.
        """

        if verbose:
            print('\nTrain DT model for tti features')

        # Create ML data
        start_time = time.time()

        data = MachineLearningData(
            input_data=input_data,
            ti_features=ALL_TI_FEATURES,
            include_close_feature=True,
            include_volume_feature=True,
            verbose=verbose).createMLData().values

        if verbose:
            print('- ml data creation time:',
                  datetime.timedelta(seconds=int(time.time() - start_time)))

        self._model_details['number_of_training_instances'] = data.shape

        # Adapt value to sklearn n_jobs values
        if pool_size is None:
            pool_size = -1

        # Split data to training and test set
        x_train, x_test, y_train, y_test = train_test_split(
            data[:, :-1], data[:, -1], test_size=0.20, train_size=None,
            random_state=None, shuffle=False, stratify=None)

        # DT model tuning, look for the best performing model
        if verbose:
            print('- searching for the best performing DT model')

        start_time = time.time()

        model_grid_search = GridSearchCV(
            estimator=DecisionTreeClassifier(),
            param_grid={
                'max_depth': [i for i in range(1, data.shape[1])] + [None]
            },
            n_jobs=pool_size,
            refit=False,
            verbose=0,
            return_train_score=True)

        model_grid_search.fit(X=data[:, :-1], y=data[:, -1])
        best_parameters = model_grid_search.best_params_

        if verbose:
            print('- model tuning finished in:',
              datetime.timedelta(seconds=int(time.time() - start_time)))
            print('- model tuning best score:', model_grid_search.best_score_)
            print('- model tuning best parameters:', best_parameters)

        # Selected DT model, calculate test score
        if verbose:
            print('- calculating training and test score for the selected ' +
                  'DT model')

        selected_model = DecisionTreeClassifier(**best_parameters)

        selected_model.fit(X=x_train, y=y_train)

        self._model_details['training_score'] = selected_model.score(
            X=x_train, y=y_train)

        self._model_details['test_score'] = selected_model.score(
            X=x_test, y=y_test)

        if verbose:
            print('- selected model training score:',
                  self._model_details['training_score'])
            print('- selected model test score:',
                  self._model_details['test_score'])

        # Selected DT model, train on full input data (final model)
        if verbose:
            print('- training selected model on the whole dataset')

        self._model = DecisionTreeClassifier(**best_parameters)

        self._model.fit(X=data[:, :-1], y=data[:, -1])

        if verbose:
            print('- model details:', self._model_details)
