import os

# Path variables to ensure that we can use the dataset files without absolute
# paths to resources
ROOT_PATH = os.path.dirname(__file__)
CLEAN_DATA = os.path.join(ROOT_PATH, "../wifi_db/clean_dataset.txt")
NOISY_DATA = os.path.join(ROOT_PATH, "../wifi_db/noisy_dataset.txt")
