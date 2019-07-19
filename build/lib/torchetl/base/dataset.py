from pathlib import Path, PosixPath
import pdb
from typing import Iterable, Sequence, List, Tuple
import re
import numpy as np
from collections import namedtuple

class BaseDataset:
    def __init__(self, 
        parent_directory: PosixPath, 
        extension: str,
        verbose: bool) -> None:
        """Base dataset to inherit from. Support for reading all files of a specific extension from a parent_directory directory
        
        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib
        extension
            The extension we want to include in our search from the parent_directory directory

        Returns
        -------
        None	
        """
        self.parent_directory = parent_directory
        self.extension = extension
        self.verbose = verbose
		
    def read_files(self) -> Iterable:
        """ Return parent_directory directory

        Parameters
        ----------
        None

        Returns
        -------
        origin
            the path to read files
        """
        try:
            p = Path(self.parent_directory)
            for file in p.rglob("*." + self.extension):
                yield file
        except:
            raise ValueError("Directory does not exist")

    def show_files(self, n: int = None) -> Sequence:
        """ Show files to consume

        Parameters
        ----------
        number_of_files_to_show
            number of files to show

        Returns
        -------
        origin
            the path to read files
        """
        files = [file for file in self.read_files()]
        print(files[:n])

    def create_dataset_array(self, labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create full dataset array from reading files. The columns are relative path to file and its label

        Parameters
        ----------
        None

        Returns
        -------
        Tuple of X and y	
        """
        Dataset = namedtuple('Dataset', ['filename', 'target'])

        target = []
        filename = []
        for absolute_path in self.read_files():
            # parts are the tuple of "/", folder, and name that makes up a directory
            number_of_parts_of_origin = len(self.parent_directory.parts)
            relative_path_with_name = absolute_path.parts[number_of_parts_of_origin:]
            # create posix path from tuple
            relative_path_with_name = Path(*relative_path_with_name)
            relative_path_without_name = relative_path_with_name.parent
            for encoded_label, label in enumerate(labels):
                if re.search(label, str(relative_path_without_name)):
                    filename.append(str(relative_path_with_name))
                    target.append(encoded_label)

        if self.verbose:
            print("Finished creating whole dataset array")

        dataset = Dataset(np.array(filename), np.array(target))
        return dataset

