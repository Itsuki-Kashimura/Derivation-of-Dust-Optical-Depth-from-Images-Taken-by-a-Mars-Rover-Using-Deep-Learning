import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F



# Custom dataset class for loading and preprocessing image data
class ImageDataSet(torch.utils.data.Dataset):


    def __init__(self, df_path='path-to-dataset'):

        # Load the processed dataset
        self.df = pd.read_csv(df_path, index_col=None, delimiter=',')

        # List of feature column names
        self.feature_labels  = [
            'zenith_solar', 'zenith_view', 'azimuth_dif', 'Ls',
            'red_value', 'green_value', 'blue_value',
            'rg_ratio', 'gb_ratio', 'rb_ratio'
        ]
        # Optical depth (target variable)
        self.od_list = self.df['od(880nm)']
        # Dictionary of feature columns
        self.feature_lists = {label: self.df[label] for label in self.feature_labels}

        # Min-max normalization for each feature (scaling to 0-1)
        self.feature_lists_normalized = {
            label: (lst - lst.min()) / (lst.max() - lst.min())
            for label, lst in self.feature_lists.items()
        }


    def __len__(self):

        # Return the number of samples in the dataset
        return len(self.df)
    

    def __getitem__(self, idx):

        # Get the target value and normalized features for a given index
        od = torch.tensor(self.od_list.iloc[idx], dtype=torch.float32)
        features = [
            torch.tensor(self.feature_lists_normalized[label].iloc[idx], dtype=torch.float32)
            for label in self.feature_labels
        ]

        return (od, *features)




# Training loop for one epoch
def train(model, train_loader, optimizer):

    model.train()  # Set model to training mode

    for batch in train_loader:

        # Move data to GPU
        od, *features_batch = [item.to('cuda', non_blocking=True) for item in batch]
        od = od.view(-1, 1)

        optimizer.zero_grad()  # Reset gradients

        output = model(*features_batch)  # Forward pass
        mse = F.mse_loss(output, od, reduction='mean')  # Compute MSE loss

        mse.backward()  # Backpropagation
        optimizer.step()  # Update weights




# Evaluate training loss and error for one epoch
def train_eval(model, train_loader, epoch, history):

    model.eval()  # Set model to evaluation mode

    mse_total = 0
    mae_total = 0
    num_batches = 0

    with torch.no_grad():

        for batch in train_loader:

            # Move data to GPU
            od, *features = [x.to('cuda', non_blocking=True) for x in batch]
            od = od.view(-1, 1)

            output = model(*features)  # Forward pass

            mse_total += F.mse_loss(output, od)  # Accumulate MSE loss
            mae_total += F.l1_loss(output, od)  # Accumulate MAE error
            num_batches += 1

    mse_avg = mse_total / num_batches  # Average MSE
    mae_avg = mae_total / num_batches  # Average MAE

    # Record metrics in history
    history['epoch'].append(epoch)
    history['train_loss'].append(mse_avg.item())
    history['train_error'].append(mae_avg.item())

    print(f'Epoch: {epoch} \n --Training-- loss: {mse_avg.item():.4f} Error: {mae_avg.item():.4f}')




# Evaluate validation loss and error for one epoch
def valid(model, valid_loader, history):

    model.eval()  # Set model to evaluation mode

    mse_total = 0
    mae_total = 0
    num_batches = 0

    with torch.no_grad():

        for batch in valid_loader:

            # Move data to GPU
            od, *features = [x.to('cuda', non_blocking=True) for x in batch]
            od = od.view(-1, 1)

            output = model(*features)  # Forward pass

            mse_total += F.mse_loss(output, od)  # Accumulate MSE loss
            mae_total += F.l1_loss(output, od)  # Accumulate MAE error
            num_batches += 1

    mse_avg = mse_total / num_batches  # Average MSE
    mae_avg = mae_total / num_batches  # Average MAE

    # Record metrics in history
    history['valid_loss'].append(mse_avg.item())
    history['valid_error'].append(mae_avg.item())

    print(f' --Validation-- loss: {mse_avg.item():.4f} Error: {mae_avg.item():.4f}\n')




# Run model on test data and return predictions
def test(model, test_loader):

    model.eval()  # Set model to evaluation mode

    predictions = []

    with torch.no_grad():

        for batch in test_loader:

            # Move features to GPU (ignore target)
            _, *features = [x.to('cuda') for x in batch]

            output = model(*features)  # Forward pass
            predictions.extend(output.cpu().numpy().flatten())  # Collect predictions

    return predictions




# Run model on test data multiple times with dropout enabled to estimate uncertainty
def test_stats(model, test_loader, sampling_num):

    model.train()  # Enable dropout for stochastic forward passes
    means = []  # List to store mean predictions
    stds = []   # List to store std of predictions

    with torch.no_grad():

        for batch in test_loader:

            # Move features to GPU (ignore target)
            _, *features = [x.to('cuda') for x in batch]

            preds = []

            for _ in range(sampling_num):

                output = model(*features)  # Forward pass
                preds.append(output.item())

            means.append(np.mean(preds))  # Mean prediction
            stds.append(np.std(preds))    # Prediction uncertainty (std)

    return means, stds




# Flexible fully-connected neural network with dropout
class neural_network(nn.Module):


    def __init__(self, n_layers, dropout_p=0.05):
        """
        Args:
            n_layers (int): Number of hidden layers (must be >= 3)
            dropout_p (float): Dropout probability for regularization
        Hidden layer sizes: [2**n_layers, 2**(n_layers-1), ..., 2**3]
        """

        super().__init__()  # Initialize base nn.Module

        # Check that n_layers is a valid integer >= 3
        if not isinstance(n_layers, int) or n_layers < 3:
            raise ValueError("n_layers must be an integer >= 3")

        # Store dropout probability for use in hidden layers
        self.dropout_p = dropout_p

        # Define hidden layer sizes
        hidden_sizes = [2 ** i for i in range(n_layers, 2, -1)]

        layers = []
        in_features = 10  # Number of input features

        # Build hidden layers with ReLU and Dropout
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_p))
            in_features = h

        # Output layer
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)


    def forward(self, solar_elevation, viewing_elevation, dif_azimuth, solar_longitude, red_value, green_value, blue_value, rg_ratio, gb_ratio, rb_ratio):

        # Concatenate all input features into a single tensor
        features = [
            solar_elevation, viewing_elevation, dif_azimuth, solar_longitude,
            red_value, green_value, blue_value, rg_ratio, gb_ratio, rb_ratio
        ]
        
        x = torch.cat([f.unsqueeze(1) if f.dim() == 1 else f for f in features], dim=1)
        return self.net(x)

