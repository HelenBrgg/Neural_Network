import utils
from datetime import datetime
import torch
import time
from torch import nn
import numpy as np
import dataset as ds
import metrics
from torch.utils.data import DataLoader
import feed_forward_neural_network as fcnn
import transformer_encoder_regression as transformer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lstm as lstm
import csv
import os
import uuid


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Hyperparams
test_size = 0.1


sec_len_output = 1

step_size = 1
target_col_name = "elevation_profile2"  # "Volume"
# "elevation_profile2"
timestamp_col = 'Zeit'  # "Date"
# should contain strings. Each string must correspond to a column name
exogenous_vars = ["Spannung_PL (2)", "Strom_PL (3)", "Drahtvorschub"]
input_variables = [target_col_name] + exogenous_vars
target_idx = 0  # index position of target in batched trg_y


hidden_dim = 64  # number of LSTM cells
# number of LSTM layers
print('' + str(datetime.now()))

model_list = ['transformer']  # 'lstm', 'fcnn',, ]


for x in range(0, 4):

    for model_name in model_list:
        if x == 0:
            train_data_1, test_data_1, val_data_1, name_dataset_train, name_dataset_test, name_dataset_val = utils.read_data(
                'Train0Anomalies', 'Test3Anomalies', timestamp_col_name=timestamp_col)

            sec_len = 202
        if x == 1:
            train_data_1, test_data_1, val_data_1, name_dataset_train, name_dataset_test, name_dataset_val = utils.read_data(
                'TrainMoreData0Anomalies', 'Test3Anomalies', timestamp_col_name=timestamp_col)

            sec_len = 202
        if x == 2:
            train_data_1, test_data_1, val_data_1, name_dataset_train, name_dataset_test, name_dataset_val = utils.read_data('Train3AnomaliesOnePerF', 'Test3Anomalies',
                                                                                                                             timestamp_col_name=timestamp_col)
            sec_len = 202
        if x == 3:
            train_data_1, test_data_1, val_data_1, name_dataset_train, name_dataset_test, name_dataset_val = utils.read_data('Train6AnomaliesTwoPerF', 'Test3Anomalies',
                                                                                                                             timestamp_col_name=timestamp_col)
            sec_len = 202
        print(name_dataset_train, name_dataset_test)
        window_size = sec_len+sec_len_output
        batch_size = 256
        if model_name == 'lstm':
            epochs = 1000
            learning_rate = 0.000001
            patience = 600
            num_layers = 2
            hidden_dim = 64
            model = lstm.SimpleLSTM(input_dim=len(exogenous_vars), hidden_dim=hidden_dim,
                                    num_layers=num_layers, output_dim=sec_len_output).to(device)
        if model_name == 'fcnn':
            epochs = 90
            learning_rate = 0.00001
            patience = 50
            model = fcnn.FCNN(sec_len, len(exogenous_vars)).to(device)
        if model_name == 'transformer':
            epochs = 200
            learning_rate = 0.000015
            patience = 80
            num_layers = 2
            num_heads = 4
            d_model = 512
            dropout = 0.1
            dim_feedforward = 512
            dropouta = 0.2

            model = transformer.TransformerEncoderRegressor(
                len(exogenous_vars), window_size-1, num_heads, num_layers, dropout, d_model, dim_feedforward, dropouta).to(device)
        iteration = str(x)
        model_name = model.__class__.__name__
        train_data_for_loop = train_data_1.copy()
        test_data_for_loop = test_data_1.copy()

        # specifically for the use case of the cone: shifting output target to right by half of the width of the cone
        train_data_for_loop.iloc[:, target_idx] = utils.prepare_elevation_profile(
            jump=sec_len/2, df=train_data_for_loop, column_to_be_shifted=target_idx)
        test_data_for_loop.iloc[:, target_idx] = utils.prepare_elevation_profile(
            jump=sec_len/2, df=test_data_for_loop, column_to_be_shifted=target_idx)

        # val_data_1.iloc[:, target_idx] = utils.prepare_elevation_profile(
        #  jump=sec_len/2, df=val_data_1, column_to_be_shifted=target_idx)
        train_scalers = {}

        for i in input_variables:
            scaler = StandardScaler()
            train_data_for_loop[i] = scaler.fit_transform(
                train_data_for_loop[i].values.reshape(-1, 1))
            test_data_for_loop[i] = scaler.transform(
                test_data_for_loop[i].values.reshape(-1, 1))
            train_scalers[i] = scaler
        # for i in input_variables:
        #    scaler = StandardScaler()
        #    val_data_1[i] = scaler.fit_transform(
        #        val_data_1[i].values.reshape(-1, 1))

        total_rows = len(train_data_for_loop)
        split_train = round(total_rows * 0.9)

    # Create separate dataframes for clarity
        training_data = train_data_for_loop.iloc[:split_train, :]
        validation_data = train_data_for_loop.iloc[split_train:, :]
    # test_data = data.iloc[split_val:, :]
        print("Length of train_data:", len(train_data_for_loop))
    # print("Length of validation_data:", len(validation_data))
        print("Length of test_data:", len(test_data_for_loop))
    # print("Length of val_data:", len(val_data))

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

        validation_indices = utils.get_indices_entire_sequence(
            data=validation_data,
            window_size=window_size,
            step_size=step_size)

        # Making instance of custom dataset class
        validation_data = ds.CustomDataset(
            data=torch.tensor(
                validation_data[input_variables].values).float(),
            target_feature=target_idx,
            indices=validation_indices,
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
        print('train.head()', train_data_for_loop.head())
        test_data_for_loop = test_data_1.copy()
        print('test_data_for_loop.head()', test_data_for_loop.head())

        training_data = DataLoader(
            training_data, batch_size, drop_last=True)
        validation_data = DataLoader(
            validation_data, batch_size, drop_last=True)
        test_data = DataLoader(test_data, batch_size, drop_last=True)

        loss_values = []
        loss_values_validation = []
        criterion = torch.nn.MSELoss().to(device)  # for regression
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_loss = float('inf')
        counter = 0
    # and load the first batch
    # data = DataLoader(data, batch_size, drop_last=True)
        print("Length of train_data:", len(training_data))
    # print("Length of validation_data:", len(validation_data))
        print("Length of test_data:", len(test_data))
        print(len(test_data))
        print(len(training_data))
    # print(len(data))

        start = time.time()
        for epoch in range(epochs):
            print(epoch)
            for i, (X, y) in enumerate(training_data):
                model.train()
                # print(epoch, i)
                X = X.to(device)
                y = y.to(device)
                # print(X.shape)
                # print(y.shape)
                output = model(X)
                print(output.shape)
                loss = criterion(output, y)
                optimizer.zero_grad()  # clear previous gradients
                loss.backward()
                optimizer.step()

                if (epoch+1) % 10 == 0:
                    print('epoch', epoch+1, 'loss= ', loss.item())

                loss_values.append(loss.item())

            # Validate the model
            model.eval()
            val_loss_total = 0
            with torch.no_grad():
                for i, (X, y) in enumerate(validation_data):
                    # Validation step, accumulating the loss in val_loss
                    X = X.to(device)
                    y = y.to(device)
                    output = model(X)
                    val_loss = criterion(output, y)
                    loss_values_validation.append(val_loss.item())
                    val_loss_total += val_loss.item()  # accumulate the batch loss

    #   Compute the average validation loss for the epoch
            average_val_loss = val_loss_total / len(validation_data)

    #   Check for loss improvement
            if average_val_loss < best_loss:
                print('##############################################################################',
                      epoch, average_val_loss)
                best_loss = average_val_loss
                counter = 0
                torch.save(model.state_dict(), 'best_model'+model_name+'.pt')
            else:
                counter += 1
            if counter >= patience:
                print("Early stopping at epoch: ",
                      epoch, "bestloss: ", best_loss)
                break
        directory = model_name+name_dataset_train + \
            name_dataset_test+iteration + str(datetime.now())
        os.mkdir(directory)

        end = time.time()
        # After training, plot the loss values
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values)
        plt.title('Loss during training')
        plt.xlabel('Batch')
        plt.ylabel('Loss value')
        plt.savefig(directory+'/loss' + model_name + name_dataset_train +
                    name_dataset_train + '.png')
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values_validation)
        plt.title('Loss during validation')
        plt.xlabel('Batch')
        plt.ylabel('Loss value')
        plt.savefig(directory+'/loss_validation' + model_name + name_dataset_train +
                    name_dataset_train + '.png')

        all_predictions = []
        all_y = []

        for i, (X, y) in enumerate(test_data):
            X = X.to(device)
            y = y.to(device)

            model.load_state_dict(torch.load('best_model'+model_name+'.pt'))
            model.eval()

            with torch.no_grad():
                output = model(X)

            output = output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            original_shape = output.shape
            predictions = output.reshape(-1, original_shape[-1])
            y = y.reshape(-1, original_shape[-1])

            # Inverse transform predictions
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
        mae = metrics.mean_absolute_error(all_predictions, all_y)
        mse = metrics.mean_squared_error(all_predictions, all_y)
        rmse = metrics.root_mean_squared_error(
            all_predictions, all_y)
        r2 = metrics.r_squared(all_predictions, all_y)

        unique_id = str(uuid.uuid4())
        model_metrics = {'unique_id': unique_id + iteration, 'model': model_name, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2':
                         r2, 'learning rate': learning_rate, 'epoch': epoch, 'patience': patience, 'duration': end-start, 'test_dataset': name_dataset_test, 'train_dataset': name_dataset_train, 'architecture': model, 'windowsize': window_size, 'batchsize': batch_size}
        with open("model_metrics"+name_dataset_train+name_dataset_test+".csv", "a",  newline="") as fp:
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
        for i, (X, y) in enumerate(training_data):
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

            # Inverse transform predictions
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

    # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(all_y_flat, label='Actual')
        plt.plot(all_predictions_flat, label='Predicted')
        plt.legend()
        plt.title('Actual vs Predicted for train batches ' +
                  model_name + name_dataset_train + name_dataset_test+unique_id)
        plt.savefig(directory+'/actual_and_predicted_train_eval' + model_name +
                    name_dataset_train + name_dataset_test + unique_id + '.png')
        all_predictions = []
        all_y = []
        for i, (X, y) in enumerate(validation_data):
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

        # Inverse transform predictions
            predictions = train_scalers[target_col_name].inverse_transform(
                predictions)
            y = train_scalers[target_col_name].inverse_transform(y)

        # Append predictions and actual values to the lists
            all_predictions.append(predictions)
            all_y.append(y)

        all_predictions = np.concatenate(all_predictions)
        all_y = np.concatenate(all_y)

    # Flatten predictions and actual values
        all_predictions_flat = all_predictions.flatten()
        all_y_flat = all_y.flatten()

    # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(all_y_flat, label='Actual')
        plt.plot(all_predictions_flat, label='Predicted')
        plt.legend()
        plt.title('Actual vs Predicted for val batches ' +
                  model_name + name_dataset_train + name_dataset_test+unique_id)
        plt.savefig(directory+'/actual_and_predicted_val_eval' + model_name +
                    name_dataset_train + name_dataset_test + unique_id + '.png')
