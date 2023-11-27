import utils

import torch
import numpy as np
import dataset as ds
import metrics
from torch.utils.data import DataLoader
import feed_forward_neural_network as fcnn
import transformer_encoder_regression as transformer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import lstm as lstm
import csv
import os
import uuid

# runs on GPU if possible
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Hyperparams
sec_len_output = 1  # length of the predicted output
step_size = 1  # size of each step as the data sequence is traversed by the moving window
target_col_name = "elevation_profile2"  # column name of the predicted output
timestamp_col = 'Zeit'  # time column
# input features, should contain strings. Each string must correspond to a column name
exogenous_vars = ["Spannung_PL (2)", "Strom_PL (3)", "Drahtvorschub"]
input_variables = [target_col_name] + exogenous_vars
target_idx = 0  # index position of target in batched trg_y
# models that will later be trained and evaluated
model_list = ['transformer', 'lstm', 'fcnn']
# read_data reads data from data file, accepts string with name of trainfile and of testfile
train_data_1, test_data_1, name_dataset_train, name_dataset_test, = utils.read_data(
    'Train0Anomalies', 'Test3Anomalies', timestamp_col_name=timestamp_col)

# Model specific hyperparameter
for model_name in model_list:
    sec_len = 202  # size of input parameters
    window_size = sec_len+sec_len_output
    batch_size = 256
    if model_name == 'lstm':
        epochs = 200
        learning_rate = 0.000001
        # patience = 600 # in case of early stopping mechanism
        num_layers = 2  # Number of LSTM layers in the network
        hidden_dim = 64  # Dimensionality of the hidden state in each LSTM cell
        model = lstm.SimpleLSTM(input_dim=len(exogenous_vars), hidden_dim=hidden_dim,
                                num_layers=num_layers, output_dim=sec_len_output).to(device)
    if model_name == 'fcnn':
        epochs = 90
        learning_rate = 0.00001
        # patience = 50 # in case of early stopping mechanism
        model = fcnn.FCNN(sec_len, len(exogenous_vars)).to(device)
    if model_name == 'transformer':
        epochs = 150
        learning_rate = 0.000015
        # patience = 50 # in case of early stopping mechanism
       # Number of transformer layers in the network
        # Increasing the number of layers can enhance the model's ability to capture hierarchical patterns in the data.
        num_layers = 2
        # Number of attention heads in the multiheadattention models
        # More heads allow the model to focus on different parts of the input sequence, aiding in capturing diverse patterns.
        num_heads = 4
        # Dimensionality of the model's hidden state and input/output
        # A higher d_model can potentially capture more complex relationships, but may require more computational resources.
        d_model = 512
        # Dropout rate applied to the output of each sub-layer
        dropout = 0.1
        # Dimensionality of the feedforward network model
        dim_feedforward = 512
        # Dropout rate for the encoder
        dropouta = 0.2
        model = transformer.TransformerEncoderRegressor(
            len(exogenous_vars), window_size-1, num_heads, num_layers, dropout, d_model, dim_feedforward, dropouta).to(device)
    model_name = model.__class__.__name__
    train_data_for_loop = train_data_1.copy()
    test_data_for_loop = test_data_1.copy()

    # specifically for the use case of the cone: shifting output target to right by half of the width of the cone
    train_data_for_loop.iloc[:, target_idx] = utils.prepare_elevation_profile(
        jump=sec_len/2, df=train_data_for_loop, column_to_be_shifted=target_idx)
    test_data_for_loop.iloc[:, target_idx] = utils.prepare_elevation_profile(
        jump=sec_len/2, df=test_data_for_loop, column_to_be_shifted=target_idx)
    train_scalers = {}
    # scale the values in all the column
    for i in input_variables:
        scaler = StandardScaler()
        train_data_for_loop[i] = scaler.fit_transform(
            train_data_for_loop[i].values.reshape(-1, 1))
        test_data_for_loop[i] = scaler.transform(
            test_data_for_loop[i].values.reshape(-1, 1))
        train_scalers[i] = scaler

    #total_rows = len(train_data_for_loop)

# Create training- and testdata, separate dataframes for clarity
    training_data = train_data_for_loop

    training_indices = utils.get_indices_entire_sequence(
        data=training_data,
        window_size=window_size,
        step_size=step_size)

# Making instance of custom dataset class
    training_data = ds.CustomDataset(
        data=torch.tensor(training_data[input_variables].values).float(),
        target_feature=target_idx,
        indices=training_indices,
        windowsize=window_size,
    )
    test_indices = utils.get_indices_entire_sequence(
        data=test_data_for_loop,
        window_size=window_size,
        step_size=step_size)

# Making instance of custom dataset class
    test_data = ds.CustomDataset(
        data=torch.tensor(
            test_data_for_loop[input_variables].values).float(),
        target_feature=target_idx,
        indices=test_indices,
        windowsize=window_size
    )
    train_data_for_loop = train_data_1.copy()
    test_data_for_loop = test_data_1.copy()

    # Batchwise Loading of training and test data
    training_data = DataLoader(
        training_data, batch_size, drop_last=True)
    test_data = DataLoader(test_data, batch_size, drop_last=True)

    loss_values = []
    # Loss-Function
    criterion = torch.nn.MSELoss().to(device)  # for regression
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    counter = 0
    # and load the first batch

    # Model training
    for epoch in range(epochs):
        print(epoch)
        for i, (X, y) in enumerate(training_data):
            model.train()
            # for gpu
            X = X.to(device)
            y = y.to(device)

            output = model(X)
            # calculate loss
            loss = criterion(output, y)

            optimizer.zero_grad()  # clear previous gradients
            loss.backward()  # backwards propagation
            optimizer.step()  # update gradients

            if (epoch+1) % 10 == 0:
                print('epoch', epoch+1, 'loss= ', loss.item())

            # for plotting later
            loss_values.append(loss.item())
    # save model
    directory = model_name+name_dataset_train + \
        name_dataset_test + str(datetime.now())
    os.mkdir(directory)

    # After training, plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values)
    plt.title('Loss during training')
    plt.xlabel('Batch')
    plt.ylabel('Loss value')
    plt.savefig(directory+'/loss' + model_name + name_dataset_train +
                name_dataset_train + '.png')
    plt.figure(figsize=(10, 6))

    plt.xlabel('Batch')
    plt.ylabel('Loss value')

    all_predictions = []
    all_y = []

    # evaluation mode
    for i, (X, y) in enumerate(test_data):
        X = X.to(device)
        y = y.to(device)

        model.eval()

        with torch.no_grad():
            output = model(X)

        output = output.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        original_shape = output.shape
        predictions = output.reshape(-1, original_shape[-1])
        y = y.reshape(-1, original_shape[-1])

        # Inverse scaled predictions
        predictions = train_scalers[target_col_name].inverse_transform(
            output)

        y = train_scalers[target_col_name].inverse_transform(y)

        # Append predictions and actual values to the lists
        all_predictions.append(predictions)
        all_y.append(y)

    all_predictions = np.concatenate(all_predictions)
    all_y = np.concatenate(all_y)

    # Flatten predictions and actual values
    all_predictions_flat = all_predictions.flatten()
    all_y_flat = all_y.flatten()

    # calculate metrics
    mae = metrics.mean_absolute_error(all_predictions, all_y)
    mse = metrics.mean_squared_error(all_predictions, all_y)
    rmse = metrics.root_mean_squared_error(
        all_predictions, all_y)
    r2 = metrics.r_squared(all_predictions, all_y)

    unique_id = str(uuid.uuid4())
    # save metrics
    model_metrics = {'unique_id': unique_id, 'model': model_name, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2':
                     r2, 'learning rate': learning_rate, 'epoch': epoch, 'test_dataset': name_dataset_test, 'train_dataset': name_dataset_train, 'architecture': model, 'windowsize': window_size, 'batchsize': batch_size, 'last_epoch': last_epoch}
    with open(metrics + "/model_metrics" + name_dataset_train + name_dataset_test + ".csv", "a",  newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=model_metrics.keys())
        # Write the data rows
        writer.writerow(model_metrics)
        fp.close()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(all_y_flat, label='Actual')
    plt.plot(all_predictions_flat, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted for test batches ' +
              model_name + name_dataset_train + name_dataset_test+unique_id)
    plt.savefig(directory+'/actual_and_predicted_test_eval' +
                model_name + name_dataset_train + name_dataset_test + unique_id + '.png')

    all_predictions = []
    all_y = []
