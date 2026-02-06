# Derivation-of-Dust-Optical-Depth-from-Images-Taken-by-a-Mars-Rover-Using-Deep-Learning

Code associated with JGR-Planets paper "Derivation of Dust Optical Depth from Images Taken by a Mars Rover Using Deep Learning".
All code are written by Itsuki Kashimura.


modules.py provides the modules for the training, validation, and test processes. It also defines the data loader and neural network architecture. In train_model.py, you can train the model defined in modules.py. Predictions on test data are generated using the model parameters optimized in test_model.py.


Datasets used for the training, validation, and test are composed of RGB radiance factors at the sky part of images and metadata associated with observed RGB images along with optical depth. The RGB image data and its metadata used are available on the Planetary Data System (PDS) Cartgraphy and Imaging Science Node (https://pds-imaging.jpl.nasa.gov/volumes/msl.html). The opaical depth data used in this study is available on Mendeley (Lemmon, 2023).
