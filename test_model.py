import time
import pandas as pd
import torch
from modules import ImageDataSet, test_stats
from modules import neural_network


def main(n_layers):

    # Set random seed for reproducibility
    torch.manual_seed(1000)

    # Define the model and load trained weights
    model = neural_network(n_layers=n_layers)
    model.load_state_dict(torch.load('path-to-trained-model'))
    model.to('cuda')

    # Load test dataset and setup data loader
    test_dataset = ImageDataSet('path-to-test-dataset')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=3, shuffle=False) # Batch_size must be set to 1.

    # Perform testing with dropout sampling
    sampling_num = 1000

    # Measure time for the test process
    torch.cuda.synchronize()
    start = time.time()

    # Test
    od_mean, od_std = test_stats(model, test_loader, sampling_num)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start
    print(f"Test completed in {elapsed_time:.2f} seconds (sampling_num={sampling_num}).")

    results = pd.DataFrame({
        'OD_mean': od_mean,
        'OD_std': od_std
    })

    results.to_csv('path-to-save-test-predictions', index=False)



if __name__ == "__main__":

    main(n_layers=5)   # Define the number of layers used in the trained model

