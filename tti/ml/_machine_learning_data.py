"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_data.py
    Implements Data related functions for the Machine Learning features of
    the tti library.
"""

import numpy as np
import pandas as pd

from ..indicators import *
from ..utils.constants import ALL_TI_FEATURES, ML_CLASSES
from ..utils.exceptions import WrongValueForInputParameter
from ..utils.exceptions import WrongTypeForInputParameter
from ..utils.exceptions import NotEnoughInputData
from ..utils.exceptions import NoFeaturesSelectedForMLData
from ..utils.exceptions import InputDataMissingForMLData
from ..utils.exceptions import NotEnoughDataForMachineLearningTraining
from ..utils.exceptions import NotEnoughDataForMachineLearningPrediction
from ..utils import fillMissingValues


class MachineLearningData:
    """
    Machine Learning Data class implementation. Creates ML data to be used for
    the ML model training. The columns can be the values of the chosen
    technical indicators (``ti_features``), the ``close`` price, and the
    ``volume`` for a equity. The exact columns are defined from the received
    input arguments. The labels are the price direction between two consecutive
    data points (future_price - current_price). The possible values of the
    labels are ``0`` for ``downward`` price direction or when price does not
    change, and ``1`` for ``upward`` price direction.

    Args:
        input_data (pandas.DataFrame): The input data. Required input columns
            are dependent from the technical indicators which will be used as
            features (see ti_features argument). The largest column set is
            ``high``, ``low``, ``open``, ``close`` and ``volume``. The index is
            of type ``pandas.DatetimeIndex``. The ``close`` column is always
            required for the calculation of the price direction (label column
            in ML Data). The ``volume`` column is required in case it is
            included also as a feature (see ``include_volume_feature``
            argument).

        ti_features (list or None, default=ALL_TI_FEATURES): List of
            indicators to be used as features for the ML data. The format of
            the list is ``[{'ti': 'indicator_class_name', 'kwargs': {...},
            ...]``. See ``ALL_TI_FEATURES`` as usage example. The default
            value is to use all the 62 indicators. If None, or empty list is
            given, then none indicator is included as feature in the created ML
            data.

        include_close_feature (bool, default=False): Indicates whether the
            ``close`` value from the input data should be included as a feature
            in the produced ML data.

        include_volume_feature (bool, default=False): Indicates whether the
            ``volume`` value from the input data should be included as a 
            feature in the produced ML data.

        verbose (bool, default=False): If set to True, processing information
            is sent to the console.

    Raises:
        WrongTypeForInputParameter: Input argument has wrong type.
        WrongValueForInputParameter: Unsupported value for input argument.
        NotEnoughInputData: Not enough data for calculating the indicator.
        TypeError: Type error occurred when validating the ``input_data``.
        ValueError: Value error occurred when validating the ``input_data``.
        NoFeaturesSelectedForMLData: No features selected for ML data.
        NotEnoughDataForMachineLearningTraining: Not enough data for ML.
        NotEnoughDataForMachineLearningPrediction: Not enough data for ML.
    """

    def __init__(self, input_data, ti_features=ALL_TI_FEATURES, 
                 include_close_feature=False, include_volume_feature=False,
                 verbose=False):

        self._input_data = fillMissingValues(input_data=input_data)
        self._ti_features = ti_features
        self._include_close_feature = include_close_feature
        self._include_volume_feature = include_volume_feature
        self._verbose = verbose

        # Context added in the _validateInputArguments
        self._indicators_set = []

        # If fails, exception is raised
        self._validateInputArguments()

        # Context is set in createMLData
        # Columns are the values of the selected ti_features, optionally the
        # close price and volume, and the labels. Rows are same as the number
        # of the input data, reduced by one (price direction calculation).
        self._ml_data = pd.DataFrame(
            data=None, columns=[], index=self._input_data.index,
            dtype=np.float64)

        # Context is set in createPredictionData
        # Columns are the values of the selected ti_features, optionally the
        # close price and volume.
        self._prediction_data = pd.DataFrame(
            data=None, columns=[], index=self._input_data.index,
            dtype=np.float64)

    def _validateInputArguments(self):
        """
        Validates the input arguments and sets the _indicators_set by
        instantiating the indicators which will be used as features.

        Raises:
            WrongTypeForInputParameter: Input argument has wrong type
            WrongValueForInputParameter: Unsupported value for input argument
            NotEnoughInputData: Not enough data for calculating the indicator
            TypeError: Type error occurred when validating the ``input_data``
            ValueError: Value error occurred when validating the ``input_data``
            NoFeaturesSelectedForMLData: No features selected for ML data.
        """

        # Verify input arguments types
        if not isinstance(self._verbose, bool):
            raise WrongTypeForInputParameter(type(self._verbose),
                                             'verbose', 'bool')

        if not isinstance(self._include_close_feature, bool):
            raise WrongTypeForInputParameter(type(self._include_close_feature),
                                             'include_close_feature', 'bool')

        if not isinstance(self._include_volume_feature, bool):
            raise WrongTypeForInputParameter(
                type(self._include_volume_feature), 'include_volume_feature',
                'bool')

        # Make columns case insensitive
        self._input_data.columns = \
            [c.lower() for c in self._input_data.columns]

        # Verify that features have been included and required data are
        # available
        if self._ti_features is None:
            self._ti_features = []

        if len(self._ti_features) == 0 and not self._include_close_feature \
                and not self._include_volume_feature:
            raise NoFeaturesSelectedForMLData(
                ti_features=self._ti_features,
                include_close_feature=self._include_close_feature,
                include_volume_feature=self._include_volume_feature)

        if 'close' not in self._input_data.columns:
            raise InputDataMissingForMLData('close')

        if self._include_volume_feature and \
                'volume' not in self._input_data.columns:
            raise InputDataMissingForMLData('volume')

        # Validate ti_features format
        supported_indicators = []
        for item in ALL_TI_FEATURES:
            supported_indicators.append(item['ti'])

        if isinstance(self._ti_features, list) and len(self._ti_features) > 0:
            for item in self._ti_features:

                for key in item.keys():
                    if key not in ['ti', 'kwargs']:
                        raise WrongTypeForInputParameter(
                            self._ti_features, 'ti_features',
                            '[{\'ti\': \'indicator_class_name\', \'kwargs\':' +
                            ' {...}, ...]')

                if 'ti' not in item.keys() or 'kwargs' not in item.keys():
                    raise WrongTypeForInputParameter(
                        self._ti_features, 'ti_features',
                        '[{\'ti\': \'indicator_class_name\', \'kwargs\':' +
                        ' {...}, ...]')

                if item['ti'] not in supported_indicators:
                    raise WrongValueForInputParameter(
                        item['ti'], 'ti_features[\'ti\']',
                        str(supported_indicators))

                if not isinstance(item['kwargs'], dict):
                    raise WrongTypeForInputParameter(
                        self._ti_features, 'ti_features',
                        '[{\'ti\': \'indicator_class_name\', \'kwargs\': ' +
                        '{...}, ...]')

        elif isinstance(self._ti_features, list) and \
                len(self._ti_features) == 0:
            pass

        else:
            raise WrongTypeForInputParameter(
                self._ti_features, 'ti_features',
                '[{\'ti\': \'indicator_class_name\', \'kwargs\': ' +
                '{...}, ...]')

        # Verify that enough input rows have been given
        if len(self._input_data.index) - 1 <= 0:
            raise NotEnoughInputData(
                '', '> 1',
                len(self._input_data.index),
                'Not enough input data. Minimum required data are '
                '(<req_data_num>), but (<data_num>) found.')

        # Construct and validate the indicators set
        if self._verbose:
            print('\nIndicators set (ML data features)')

        for indicator in self._ti_features:

            if self._verbose:
                print('- create indicator instance:', indicator)

            self._indicators_set.append(eval(indicator['ti'])(
                input_data=self._input_data, **indicator['kwargs']))

    def createMLData(self):
        """
        Constructs data to be used for the ML model creation.

        Returns:
            pandas.DataFrame: The created ML data.

        Raises:
            NotEnoughDataForMachineLearningTraining: Not enough data for ML.
        """

        if self._verbose:
            print('\nCreate ML Data')

        # Minimum required number of input data for ML training under the
        # current implemented setup.
        if len(self._input_data.index) < 60:
            raise NotEnoughDataForMachineLearningTraining(
                len(self._input_data.index), 60)

        # Add features column
        for indicator, feature in zip(self._indicators_set, self._ti_features):
            feature_data = indicator.getTiData()
            # Because there are some inf values
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            if self._verbose:
                print('- adding feature: ', feature['ti'], ', columns: ',
                      str([feature['ti'] + '_' + c
                           for c in feature_data.columns]), sep='')

            for c in feature_data.columns:
                self._ml_data[feature['ti'] + '_' + c] = feature_data[[c]]

        if self._include_close_feature:
            self._ml_data['close'] = self._input_data[['close']]

        if self._include_volume_feature:
            self._ml_data['volume'] = self._input_data[['volume']]

        # Add label column
        self._ml_data['label'] = np.roll(
            a=self._input_data['close'].values, shift=-1, axis=0
        ) - self._input_data['close'].values

        self._ml_data.loc[
            self._ml_data.label > 0, 'label'] = ML_CLASSES['UP']
        self._ml_data.loc[
            self._ml_data.label <= 0, 'label'] = ML_CLASSES['DOWN']

        self._ml_data['label'] = self._ml_data['label'].apply(lambda x: int(x))

        # Remove last row, since it cannot include a label. Future value is not
        # known
        self._ml_data = self._ml_data.iloc[:-1, :]

        # Fill missing values
        self._ml_data = fillMissingValues(input_data=self._ml_data)

        return self._ml_data

    def createPredictionData(self):
        """
        Constructs data to be used in model prediction (input features).

        Returns:
            pandas.DataFrame: The created input features data.

        Raises:
            NotEnoughDataForMachineLearningPrediction: Not enough data for ML.
        """

        if self._verbose:
            print('\nCreate Prediction (Input Features) Data')

        # Minimum required number of input data for ML prediction under the
        # current implemented setup.
        if len(self._input_data.index) < 60:
            raise NotEnoughDataForMachineLearningPrediction(
                len(self._input_data.index), 60)

        # Add features column
        for indicator, feature in zip(self._indicators_set, self._ti_features):
            feature_data = indicator.getTiData()
            # Because there are some inf values
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            if self._verbose:
                print('- adding feature: ', feature['ti'], ', columns: ',
                      str([feature['ti'] + '_' + c
                           for c in feature_data.columns]), sep='')

            for c in feature_data.columns:
                self._prediction_data[
                    feature['ti'] + '_' + c] = feature_data[[c]]

        if self._include_close_feature:
            self._prediction_data['close'] = self._input_data[['close']]

        if self._include_volume_feature:
            self._prediction_data['volume'] = self._input_data[['volume']]

        # Fill missing values
        self._prediction_data = fillMissingValues(
            input_data=self._prediction_data)

        return self._prediction_data
