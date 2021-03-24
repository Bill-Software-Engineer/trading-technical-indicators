"""
Trading-Technical-Indicators (tti) python library

File name: example_machine_learning_mlp_loaded_model.py
    Example code for the trading technical indicators library, Machine Learning
    features (Load trained MLP model).
"""

import pandas as pd
from tti.ml import MachineLearningLoadedModel

# Read data from csv file. Set the index to the correct column (dates column)
input_data = pd.read_csv(
    './data/SCMN.SW.csv', parse_dates=True, index_col=0)

# Load trained model
loaded_model = MachineLearningLoadedModel()
loaded_model.mlLoadModel(file_name='./data/mlp_trained_model_SCMN_SW.dmp')

# Get loaded model details
print('\nLoaded MLP model details:', loaded_model.mlModelDetails())

# Use loaded model to make predictions (minimum data required are the last
# 60 periods, prediction is for the next period).
print('\nModel prediction:',
      loaded_model.mlPredict(input_data=input_data.iloc[-60:, :]))
