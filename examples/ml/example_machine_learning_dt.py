"""
Trading-Technical-Indicators (tti) python library

File name: example_machine_learning_dt.py
    Example code for the trading technical indicators library, Machine Learning
    features (DT model).
"""

import pandas as pd
from tti.ml import MachineLearningDT

# Read data from csv file. Set the index to the correct column (dates column)
input_data = pd.read_csv(
    './data/SCMN.SW.csv', parse_dates=True, index_col=0)

# Train a Decision Tree (DT) model
model = MachineLearningDT()

model.mlTrainModel(
    input_data=input_data, pool_size=6, verbose=False)

# Get trained model details
print('\nTrained DT model details:', model.mlModelDetails())

# Predict (use 60 last periods to predict next period)
print('\nModel prediction:',
      model.mlPredict(input_data=input_data.iloc[-60:, :]))

# Save model
model.mlSaveModel(file_name='./data/dt_trained_model_SCMN_SW.dmp')
