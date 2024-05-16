import numpy as np
import torch

# Logging configuration
LOGGING_PATH = "logs"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configurations
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 1000
PATIENCE = 10
NUM_CLASS = 10

# Attack configurations
NORM = np.inf
ATTACK_ITER = 10

# Analysis configuration
STD_DEV_MULTIPLIER = 3

# Configurations for Gaussian Mixture Model
MAX_PEAKS = 10
GMM_MAX_ITER = 1000
GMM_N_INIT = 10
THRESHOLD_MULTIPLIER = 3
NUM_SAMPLES = 100
