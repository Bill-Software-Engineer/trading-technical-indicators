"""
Trading-Technical-Indicators (tti) python library

File name: _machine_learning_loaded_model.py
    Implements Loaded Trained Model interface for the Machine Learning features
    of the tti library.
"""

from ._machine_learning_api import MachineLearningAPI
from ..utils.exceptions import ModelTrainingIsNotSupported


class MachineLearningLoadedModel(MachineLearningAPI):
    """
    Implements an interface for Loading trained models.
    """

    def __init__(self):

        super().__init__(model_type='Loaded Trained Model')

    def mlTrainModel(self):
        """
        Not supported for loaded models.

        Raises:
            ModelTrainingIsNotSupported: Model training is not supported.
        """

        raise ModelTrainingIsNotSupported()
