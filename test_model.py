import time
import pandas as pd
import torch
from modules import ImageDataSet, test_stats
from modules import neural_network


def main(n_layers):
    """
    Main function to test the trained neural network model.
    Args:
        n_layers (int): Number of layers in the neural network.
    """

    # Set random seed for reproducibility
    torch.manual_seed(1000)

    # Define the model and load trained weights
    model = neural_network(n_layers=n_layers).to('cuda')  # Instantiate the model and move to GPU
    model.load_state_dict(torch.load('path-to-trained-model'))  # Load trained weights

    # Load test dataset and setup data loader
    test_dataset = ImageDataSet('path-to-test-dataset')
    # Batch_size must be set to 1 for uncertainty estimation
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=3, shuffle=False)

    # Perform testing with dropout sampling for uncertainty estimation
    # Used to estimate prediction uncertainty by checking the variation in outputs
    sampling_num = 1000    # How many times to run the model on the same input with dropout enabled

    # Measure time for the test process
    torch.cuda.synchronize()
    start = time.time()  # Start timer

    # Test the model and obtain mean and std of predictions
    od_mean, od_std = test_stats(model, test_loader, sampling_num)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start  # Calculate elapsed time
    print(f"Test completed in {elapsed_time:.2f} seconds (sampling_num={sampling_num}).")

    # Save test results (mean and std) to CSV file
    results = pd.DataFrame({
        'OD_mean': od_mean,
        'OD_std': od_std
    })
    results.to_csv('path-to-save-test-predictions', index=False)



if __name__ == "__main__":

    main(n_layers=5)   # Test the model with the same number of layers as used in training

