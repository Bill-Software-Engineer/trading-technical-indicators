"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_api.py
    Defines the API for the Machine Learning features of the tti library.
"""

import joblib

from abc import ABC, abstractmethod
from ..utils.constants import ML_CLASSES
from ..utils.constants import ALL_TI_FEATURES
from ._machine_learning_data import MachineLearningData


class MachineLearningAPI(ABC):
    """
    Machine Learning API class implementation.

    Args:
        model_type (str): The model type of the trained model (i.e. MLP, DT).
            It is set by the child class.

    Attributes:
        _model (object): The trained model object.

        _scaler (sklearn.preprocessing.StandardScaler): Scaler used with the
            training data.

        _model_details (dict): Dictionary with model details. It contains the
            ``model_type`` see model_type argument, ``training_score`` the
            score of the trained model on the training data, ``test_score`` the
            score of the trained model on the test data,
            ``number_of_training_instances`` the number of instances used to
            train the model, ``classes`` a definition (dict) of the classes,
            ``scaler_used`` whether a scaler used with the training data, and
            ``dump_file`` the file where the model is saved.
    """
    
    def __init__(self, model_type):

        # Trained model, when None there is not any trained model yet
        self._model = None

        # Data scaler, when None there is not any scaler used
        self._scaler = None

        # Training score, test score, and number of training instances are set
        # by the child class
        self._model_details = {'model_type': model_type,
                               'training_score': None,
                               'test_score': None,
                               'number_of_training_instances': None,
                               'classes': ML_CLASSES,
                               'scaler_used': False,
                               'dump_file': None}

    def mlSaveModel(self, file_name):
        """
        Saves the trained model to a file.

        Args:
            file_name (str): The file where the serialized trained model should
                be saved.
        """

        self._model_details['dump_file'] = file_name

        self._model_details['scaler_used'] = False if self._scaler is None \
            else True

        with open(file_name, 'wb') as out_file:
            joblib.dump((self._scaler, self._model, self._model_details),
                        out_file)

    def mlLoadModel(self, file_name):
        """
        Loads a trained model from a file.

        Args:
            file_name (str): The file from where the trained model should be
                deserialized.
        """

        with open(file_name, 'rb') as in_file:
            self._scaler, self._model, self._model_details = \
                joblib.load(in_file)

    def mlModelDetails(self):
        """
        Returns details about the trained model.

        Returns:
            dict: The model details dictionary. See class attributes for
            details.
        """

        self._model_details['scaler_used'] = False if self._scaler is None \
            else True

        return self._model_details

    @abstractmethod
    def mlTrainModel(self, **kwargs):
        """
        Trains a machine learning model for prices direction predictions.
        """

        raise NotImplementedError

    def mlPredict(self, input_data):
        """
        Returns a prediction.

        Args:
            input_data (pandas.DataFrame): The input data. Required input
                columns are ``high``, ``low``, ``open``, ``close`` and
                ``volume``. The index is of type ``pandas.DatetimeIndex``. The
                minimum number of data required for prediction is 60 periods.

        Returns:
            (str, int): The class predicted by the trained model. Possible
            return values are ('DOWN', 0) and ('UP', 1).
        """

        # Calculate input features for the prediction of the next period
        data = MachineLearningData(
            input_data=input_data,
            ti_features=ALL_TI_FEATURES,
            include_close_feature=True,
            include_volume_feature=True,
            verbose=False).createPredictionData().values[-1, :].reshape(1, -1)

        if self._scaler is not None:
            data = self._scaler.transform(X=data)

        prediction = self._model.predict(X=data)

        for k, v in ML_CLASSES.items():
            if v == prediction:
                return k, v
