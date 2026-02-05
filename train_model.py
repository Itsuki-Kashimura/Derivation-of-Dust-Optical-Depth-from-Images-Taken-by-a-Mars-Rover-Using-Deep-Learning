import time
import pandas as pd
import torch
import torchinfo
from torch.optim.lr_scheduler import OneCycleLR
from modules import ImageDataSet, train, train_eval, valid
from modules import neural_network



def main(n_layers):

    # Set random seed for reproducibility
    torch.manual_seed(1000)

    # Define the model
    model = neural_network(n_layers=n_layers).to('cuda')

    # Load datasets
    train_dataset = ImageDataSet(df_path="path-to-training-dataset")
    valid_dataset = ImageDataSet(df_path="path-to-validation-dataset")

    # Setup data loaders
    loader_args = dict(batch_size=4, num_workers=2, pin_memory=True, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_args)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **loader_args)

    # Set up parameters for training
    epochs = 500
    lr = 1e-3
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=10*lr, epochs=epochs, steps_per_epoch=len(train_loader))

    # History dictionary to store loss values
    history = {
        'epoch': [],
        'train_loss': [],
        'train_error': [],
        'valid_loss': [],
        'valid_error': []
    }

    # Print parameters for training
    print("Epochs: ", epochs)
    print("Learning rate: ", lr)
    print("Optimizer: Adam")
    print("Scheduler: OneCycleLR")
    print("Model Summary:")
    torchinfo.summary(
        model,
        input_size=[(100, 1)] * 10
    )

    print(
        '\n\n\n' +
        '~' * 50 +
        '\nLoss and error in each epoch\n' +
        '~' * 50 +
        '\n'
    )


    # Measure time for the training and validation process
    torch.cuda.synchronize()
    start = time.time()

    # Training and Validation
    for epoch in range(1, epochs+1):

        train(model, train_loader, optimizer)
        train_eval(model, train_loader, epoch, history)
        valid(model, valid_loader, history)

        scheduler.step()

    torch.cuda.synchronize()
    elapsed_time = time.time() - start
    print(f"Training and validation completed in {elapsed_time:.2f} seconds.")


    pd.DataFrame(history).to_csv('path-to-save-history', index=False)
    torch.save(model.state_dict(), 'path-to-save-model')



if __name__ == '__main__':

    main(n_layers=5)   # Define number of layers

