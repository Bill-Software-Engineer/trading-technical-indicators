tti.ml usage examples
=====================

Below is an example on how to use the ``tti.ml`` package API.

.. code-block:: bash
    :caption: Input data (part of them)

                    high         low        open       close     volume
    Date
    1998-10-05  388.084991  342.890991  343.872986  369.908997  2732524.0
    1998-10-06  385.628998  371.382996  372.856995  385.138000   477160.0
    1998-10-07  403.313995  377.769012  383.173004  398.402008   647460.0
    1998-10-08  396.928009  376.295013  390.049988  382.190002   292460.0
    1998-10-09  394.963013  386.119995  392.015015  388.084991   259155.0
                    ...         ...         ...         ...        ...


.. code-block:: python
    :caption: MLP model usage example

    """
    Trading-Technical-Indicators (tti) python library

    File name: example_machine_learning_mlp.py
        Example code for the trading technical indicators library, Machine Learning
        features (MLP model).
    """

    import pandas as pd
    from tti.ml import MachineLearningMLP

    # Read data from csv file. Set the index to the correct column (dates column)
    input_data = pd.read_csv(
        './data/SCMN.SW.csv', parse_dates=True, index_col=0)

    # Train a Multilayer Perceptron (MLP) model
    model = MachineLearningMLP()

    model.mlTrainModel(
        input_data=input_data, pool_size=6, verbose=False)

    # Get trained model details
    print('\nTrained MLP model details:', model.mlModelDetails())

    # Predict (use 60 last periods to predict next period)
    print('\nModel prediction:',
        model.mlPredict(input_data=input_data.iloc[-60:, :]))

    # Save model
    model.mlSaveModel(file_name='./data/mlp_trained_model_SCMN_SW.dmp'

.. code-block:: bash
    :caption: Output of the MLP model usage example

    Trained MLP model details: {'model_type': 'MLP', 'training_score': 0.7300884955752213, 'test_score': 0.6787610619469027, 'number_of_training_instances': (5650, 86), 'classes': {'DOWN': 0, 'UP': 1}, 'scaler_used': True, 'dump_file': None}

    Model prediction: ('DOWN', 0)

.. code-block:: python
    :caption: DT model usage example

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

.. code-block:: bash
    :caption: Output of the DT model usage example

    Trained DT model details: {'model_type': 'DT', 'training_score': 0.738495575221239, 'test_score': 0.7309734513274336, 'number_of_training_instances': (5650, 86), 'classes': {'DOWN': 0, 'UP': 1}, 'scaler_used': False, 'dump_file': None}

    Model prediction: ('DOWN', 0)

.. code-block:: python
    :caption: Load trained model usage example

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

.. code-block:: bash
    :caption: Output of the DT model usage example

    Loaded DT model details: {'model_type': 'DT', 'training_score': 0.738495575221239, 'test_score': 0.7309734513274336, 'number_of_training_instances': (5650, 86), 'classes': {'DOWN': 0, 'UP': 1}, 'scaler_used': False, 'dump_file': './data/dt_trained_model_SCMN_SW.dmp'}

    Model prediction: ('DOWN', 0)
