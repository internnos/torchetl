from torchetl.etl import Extract
from pathlib import Path
import pandas as pd

parent_directory = 'data'

combined_dataset = Extract(parent_directory = parent_directory, 
    extension = 'jpg', 
    labels = ['attack', 'real'], 
    train_size = 0.8,
    random_state = 69,
    verbose = True
)

combined_dataset.extract(file_prefix='exp', save_path='data', is_random_sampling=False)