import utils
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
batch_size = 128
sec_len_output = 1
sec_len = 200  # must be even
window_size = sec_len+sec_len_output
step_size = 1
target_col_name = "elevation_profile2"  # "Volume"
# "elevation_profile2"
timestamp_col = 'Zeit'  # "Date"
# should contain strings. Each string must correspond to a column name
exogenous_vars = ["Spannung_PL (2)", "Strom_PL (3)", "Drahtvorschub",
                  "Robotergeschwindigkeit", "ZerstÃ¤ubergasmenge"]
input_variables = [target_col_name] + exogenous_vars
target_idx = 0  # index position of target in batched trg_y

epochs = 10
learning_rate = 0.000001

data, name_dataset = utils.read_data(timestamp_col_name=timestamp_col)

# specifically for the use case of the cone
data.iloc[:, target_idx] = utils.prepare_elevation_profile(
    jump=sec_len/2, df=data, column_to_be_shifted=target_idx)
scalers = {}
for i in input_variables:
    scaler = StandardScaler()
    data[i] = scaler.fit_transform(data[i].values.reshape(-1, 1))
    scalers[i] = scaler
split_train = round(len(data) * 0.7)
split_val = split_train + round(len(data) * 0.15)

# Split the data
training_data = data.iloc[:split_train, :]
validation_data = data.iloc[split_train:split_val, :]
test_data = data.iloc[split_val:, :]


training_indices = utils.get_indices_entire_sequence(
    data=training_data,
    window_size=window_size,
    step_size=step_size)
print('training_indices[0]', training_indices[0], training_indices[1])

# Making instance of custom dataset class
training_data = ds.CustomDataset(
    data=torch.tensor(training_data[input_variables].values).float(),
    target_feature=target_idx,
    indices=training_indices,
    windowsize=window_size,
)
indices = utils.get_indices_entire_sequence(
    data=data,
    window_size=window_size,
    step_size=step_size)
print('training_indices[0]', training_indices[0], training_indices[1])

# Making instance of custom dataset class
data = ds.CustomDataset(
    data=torch.tensor(data[input_variables].values).float(),
    target_feature=target_idx,
    indices=indices,
    windowsize=window_size

)

test_indices = utils.get_indices_entire_sequence(
    data=test_data,
    window_size=window_size,
    step_size=step_size)

# Making instance of custom dataset class
test_data = ds.CustomDataset(
    data=torch.tensor(test_data[input_variables].values).float(),
    target_feature=target_idx,
    indices=test_indices,
    windowsize=window_size
)

training_data = DataLoader(training_data, batch_size, drop_last=True)
# validation_data = DataLoader(validation_data, batch_size, drop_last=True)
test_data = DataLoader(test_data, batch_size, drop_last=True)
data = DataLoader(data, batch_size, drop_last=True)

print(len(test_data))
print(len(training_data))
print(len(data))

num_heads = 4
num_layers = 2


hidden_dim = 32  # number of LSTM cells
num_layers = 2  # number of LSTM layers

model = lstm.SimpleLSTM(input_dim=len(exogenous_vars), hidden_dim=hidden_dim,
                        num_layers=num_layers, output_dim=sec_len_output).to(device)
# model = fcnn.FCNN(sec_len, len(exogenous_vars)).to(device)
# model = transformer.TransformerEncoderRegressor(
# len(exogenous_vars), window_size-1, num_heads, num_layers).to(device)

criterion = torch.nn.MSELoss().to(device)  # for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_values = []
patience = 5
best_loss = float('inf')
counter = 0
# and load the first batch
start = time.time()
for epoch in range(epochs):
    for i, (X, y) in enumerate(training_data):
        model.train()
        print(epoch, i)
        X = X.to(device)
        y = y.to(device)
        print(X.shape)
        print(y.shape)
        output = model(X)
        print(output.shape)
        loss = criterion(output, y)
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()
        optimizer.step()
        # running_loss += loss.item() * X.size(0)
        # train_loss = running_loss / len(training_data)
        if (epoch+1) % 10 == 0:
            print('epoch', epoch+1, 'loss= ', loss.item())
        # print(prediction, trg_y)
        # print('Batch', i+1, 'Loss:', loss.item())
        loss_values.append(loss.item())  # Append the loss value to the list

    # Validate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(validation_data):
            # Validation step, accumulating the loss in val_loss
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            val_loss = criterion(output, y)

    # Check for loss improvement
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping at epoch: ", epoch, "bestloss: ", best_loss)
            break

end = time.time()
# After training, plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(loss_values)
plt.title('Loss during training')
plt.xlabel('Batch')
plt.ylabel('Loss value')
plt.savefig('loss' + model + name_dataset + '.png')

all_predictions = []
all_y = []
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

    # Inverse transform predictions
    predictions = scalers[target_col_name].inverse_transform(output)
    # predictions = scaler.inverse_transform(predictions)
    y = scalers[target_col_name].inverse_transform(y)
    # trg_y = scaler.inverse_transform(trg_y)

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
rmse = metrics.root_mean_squared_error(all_predictions, all_y)
r2 = metrics.r_squared(all_predictions, all_y)

model_metrics = {'model': model, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2':
                 r2, 'learning rate': learning_rate, 'duration': end-start, 'dataset': name_dataset}
with open("model_metrics.csv", "w", newline="") as fp:
    # Create a writer object
    writer = csv.DictWriter(fp, fieldnames=model_metrics.keys())

    # Write the data rows
    writer.writerow(model_metrics)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(all_y_flat, label='Actual')
plt.plot(all_predictions_flat, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted for test batches')
plt.savefig('actual_and_predicted_test_eval' + model + name_dataset + '.png')

all_predictions = []
all_y = []
for i, (X, y) in enumerate(data):
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
    predictions = scalers[target_col_name].inverse_transform(output)
    # predictions = scaler.inverse_transform(predictions)
    y = scalers[target_col_name].inverse_transform(y)
    # trg_y = scaler.inverse_transform(trg_y)

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
plt.title('Actual vs Predicted for all batches')
plt.savefig('actual_and_predicted_all_eval' + model + name_dataset + '.png')
