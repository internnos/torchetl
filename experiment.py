from pathlib import Path
from torchvision import transforms
from torchetl.etl import ExtractTwoPartitions, TransformAndLoad
import pandas as pd

parent_directory = Path.cwd() / 'data' / 'replay-attack-style' 
print(parent_directory)

combined_dataset = ExtractTwoPartitions(parent_directory = parent_directory, 
              extension = 'jpg', 
              labels = ['attack', 'real'], 
              train_size = 0.8,
              random_state = 69,
              verbose = True,
            )

combined_dataset.extract("AFAD", parent_directory, False)