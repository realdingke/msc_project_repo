import numpy as np


# Use these for importing the dataset from the text files
# ...or don't, it's your choice

# load_data(paths.CLEAN_DATA)
# load_data(paths.NOISY_DATA)
def load_data(file):
    return np.loadtxt(file)
