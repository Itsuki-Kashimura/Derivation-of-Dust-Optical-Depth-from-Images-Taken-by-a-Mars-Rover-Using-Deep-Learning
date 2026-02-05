import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F



class ImageDataSet(torch.utils.data.Dataset):


    def __init__(self, df_path='path-to-dataset'):
       
        # load the processed dataset
        self.df = pd.read_csv(df_path, index_col=None, delimiter=',')

        self.feature_labels  = [
            'zenith_solar', 'zenith_view', 'azimuth_dif', 'Ls',
            'red_value', 'green_value', 'blue_value',
            'rg_ratio', 'gb_ratio', 'rb_ratio'
        ]
        self.od_list = self.df['od(880nm)']
        self.feature_lists = {label: self.df[label] for label in self.feature_labels}

        # Scaling input variables to 0-1 by min-max normalization
        self.feature_lists_normalized = {
            label: (list - list.min()) / (list.max() - list.min())
            for label, list in self.feature_lists.items()
        }


    def __len__(self):
        
        return len(self.df)
    

    def __getitem__(self, idx):
        
        od = torch.tensor(self.od_list.iloc[idx], dtype=torch.float32)
        features = [
            torch.tensor(self.feature_lists_normalized[label].iloc[idx], dtype=torch.float32)
            for label in self.feature_labels
        ]
 
        return (od, *features)



def train(model, train_loader, optimizer):

    model.train()

    for batch in train_loader:

        od, *features_batch = [item.to('cuda', non_blocking=True) for item in batch]
        od = od.view(-1, 1)

        optimizer.zero_grad()

        output = model(*features_batch)
        mse = F.mse_loss(output, od, reduction='mean')

        mse.backward()
        optimizer.step()



def train_eval(model, train_loader, epoch, history):

    model.eval()

    mse_total = 0
    mae_total = 0
    num_batches = 0

    with torch.no_grad():

        for batch in train_loader:

            od, *features = [x.to('cuda', non_blocking=True) for x in batch]
            od = od.view(-1, 1)

            output = model(*features)

            mse_total += F.mse_loss(output, od)
            mae_total += F.l1_loss(output, od)
            num_batches += 1

    mse_avg = mse_total / num_batches
    mae_avg = mae_total / num_batches

    history['epoch'].append(epoch)
    history['train_loss'].append(mse_avg.item())
    history['train_error'].append(mae_avg.item())

    print(f'Epoch: {epoch} \n --Training-- loss: {mse_avg.item():.4f} Error: {mae_avg.item():.4f}')



def valid(model, valid_loader, history):

    model.eval()

    mse_total = 0
    mae_total = 0
    num_batches = 0

    with torch.no_grad():

        for batch in valid_loader:

            od, *features = [x.to('cuda', non_blocking=True) for x in batch]
            od = od.view(-1, 1)
            output = model(*features)

            mse_total += F.mse_loss(output, od)
            mae_total += F.l1_loss(output, od)
            num_batches += 1

    mse_avg = mse_total / num_batches
    mae_avg = mae_total / num_batches

    history['valid_loss'].append(mse_avg.item())
    history['valid_error'].append(mae_avg.item())

    print(f' --Validation-- loss: {mse_avg.item():.4f} Error: {mae_avg.item():.4f}\n')



def test(model, test_loader):

    model.eval()

    predictions = []

    with torch.no_grad():

        for batch in test_loader:

            _, *features = [x.to('cuda') for x in batch]

            output = model(*features)
            predictions.extend(output.cpu().numpy().flatten())

    return predictions



def test_stats(model, test_loader, sampling_num):

    model.train()
    means = []
    stds = []

    with torch.no_grad():

        for batch in test_loader:

            _, *features = [x.to('cuda') for x in batch]

            preds = []

            for _ in range(sampling_num):

                output = model(*features)
                preds.append(output.item())

            means.append(np.mean(preds))
            stds.append(np.std(preds))

    return means, stds



class neural_network(nn.Module):


    def __init__(self, n_layers, dropout_p=0.05):
        """
        n_layers: int >= 3
        Number of neurons in hidden layers : [2**n_layers, 2**(n_layers-1), ..., 2**3]
        """
        super().__init__()

        if not isinstance(n_layers, int) or n_layers < 3:
            raise ValueError("n_layers must be an integer >= 3")

        self.dropout_p = dropout_p

        hidden_sizes = [2 ** i for i in range(n_layers, 2, -1)]

        layers = []
        in_features = 10

        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_p))
            in_features = h

        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)


    def forward(self, solar_elevation, viewing_elevation, dif_azimuth, solar_longitude, red_value, green_value, blue_value, rg_ratio, gb_ratio, rb_ratio):
        
        features = [
            solar_elevation, viewing_elevation, dif_azimuth, solar_longitude,
            red_value, green_value, blue_value, rg_ratio, gb_ratio, rb_ratio
        ]
        
        x = torch.cat([f.unsqueeze(1) if f.dim() == 1 else f for f in features], dim=1)
        
        return self.net(x)

