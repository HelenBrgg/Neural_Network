import utils
import torch
from torch import nn
import numpy as np
import dataset as ds
from torch.utils.data import DataLoader
import neural_network
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
window_size = 200
step_size = 1
target_col_name = "elevation_profile2"
timestamp_col = "Zeit"
exogenous_vars = ["Spannung_PL (2)", "Strom_PL (3)", "Drahtvorschub"]
# "Robotergeschwindigkeit", "ZerstÃ¤ubergasmenge"]  # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0  # index position of target in batched trg_y


epochs = 10
learning_rate = 0.001


data = utils.read_data(timestamp_col_name=timestamp_col)
scalers = {}
for i in input_variables:
    scaler = StandardScaler()
    data[i] = scaler.fit_transform(data[i].values.reshape(-1, 1))
    scalers[i] = scaler
split = round(len(data)*0.9)

training_data = data.iloc[:split, :]
test_data = data.iloc[len(training_data):, :]
print(data)

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

training_data = DataLoader(training_data, batch_size, drop_last=False)
test_data = DataLoader(test_data, batch_size, drop_last=False)

model = neural_network.FCNN(window_size, len(exogenous_vars)).to(device)
criterion = torch.nn.MSELoss()  # for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_values = []
# and load the first batch
for epoch in range(epochs):
    for i, (X, y) in enumerate(training_data):
        print(type(X))
        print(X)
        print(type(y))
        print(y)
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print('epoch', epoch+1, 'loss= ', loss.item())
        # print(prediction, trg_y)
        # print('Batch', i+1, 'Loss:', loss.item())
        loss_values.append(loss.item())  # Append the loss value to the list

# After training, plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(loss_values)
plt.title('Loss during training')
plt.xlabel('Batch')
plt.ylabel('Loss value')
plt.savefig('loss.png')

all_predictions = []
all_trg_y = []
for i, (X, y) in enumerate(test_data):
    model.eval()

    with torch.no_grad():
        output = model(X)

    #predictions = output.detach().cpu().numpy()
    #trg_y = trg_y.detach().cpu().numpy()

    original_shape = output.shape
    predictions = output.reshape(-1, original_shape[-1])
    y = y.reshape(-1, original_shape[-1])

    # Inverse transform predictions
    predictions = scalers["elevation_profile2"].inverse_transform(output)
    # predictions = scaler.inverse_transform(predictions)
    trg_y = scalers["elevation_profile2"].inverse_transform(trg_y)
    # trg_y = scaler.inverse_transform(trg_y)

    # Append predictions and actual values to the lists
    all_predictions.append(predictions)
    all_trg_y.append(trg_y)

all_predictions = np.concatenate(all_predictions)
all_trg_y = np.concatenate(all_trg_y)

# Flatten predictions and actual values
all_predictions_flat = all_predictions.flatten()
all_trg_y_flat = all_trg_y.flatten()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(all_trg_y_flat, label='Actual')
plt.plot(all_predictions_flat, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted for all batches')
plt.savefig('actual_and_predicted_test_eval.png')
