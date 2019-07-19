from pathlib import Path
from torchvision import transforms
from torchetl.etl import ExtractTwoPartitions, TransformAndLoad, Extract
from torchetl.base.dataset import BaseDataset
import pandas as pd
from typing import List
from pathlib import PosixPath
import pdb

dataset = BaseDataset(Path.cwd() / 'data' / 'replay-attack-style', "jpg", ['attack', 'real'], True)
pdb.set_trace()
data = dataset.create_dataset_array(['attack', 'real'])


# class IMDB(ExtractTwoPartitions):
#     def __init__(self, 
#                 parent_directory: PosixPath, 
#                 extension: str,
#                 labels: List[str], 
#                 train_size: float, 
#                 random_state: int, 
#                 verbose: bool) -> None:
#         """Class for creating csv files of train, validation, and test
#         Parameters
#         ----------
#         parent_directory
#             The parent_directory folder path. It is highly recommended to use Pathlib. Use full path
#             instead of relative path
#         extension
#             The extension we want to include in our search from the parent_directory
#         labels
#         Returns
#         -------
#         None	
#         """
#         super().__init__(parent_directory, extension, labels, train_size, random_state, verbose)

# parent_directory = Path.cwd() / 'data' / 'replay-attack-style' 
# print(parent_directory)

# combined_dataset = BaseDataset(
#                             parent_directory = parent_directory, 
#                             extension = 'jpg', 
#                             labels = ['attack', 'real'], 
#                             verbose = True,
# )



# X,y = combined_dataset.create_dataset_array(labels=['male', 'female'])    
# combined_dataset = ExtractTwoPartitions(parent_directory = parent_directory, 
#               extension = 'jpg', 
#               labels = ['attack', 'real'], 
#               train_size = 0.8,
#               random_state = 69,
#               verbose = True,
#             )

# combined_dataset.extract("combined", parent_directory, False)