# Derivation-of-Dust-Optical-Depth-from-Images-Taken-by-a-Mars-Rover-Using-Deep-Learning

Code associated with JGR-Planets paper "Derivation of Dust Optical Depth from Images Taken by a Mars Rover Using Deep Learning".
All code are written by Itsuki Kashimura.


modules.py provides the modules for the training, validation, and test processes. It also defines the data loader and neural network architecture. In train_model.py, you can train the model defined in modules.py. Predictions on test data are generated using the model parameters optimized in test_model.py.
