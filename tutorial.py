from torchetl.etl import Extract
from pathlib import Path
import pandas as pd

parent_directory = Path.cwd() / 'data'

combined_dataset = Extract(parent_directory = parent_directory, 
              extension = 'jpg', 
              labels = ['attack', 'real'], 
              train_size = 0.8,
              random_state = 69,
              verbose = True,
            )
import pdb; pdb.set_trace()
combined_dataset._create_dataset_array()