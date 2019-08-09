from pathlib import Path, PosixPath
import pdb
from typing import Iterable, Sequence, List, Tuple
import re
import numpy as np
from collections import namedtuple
from typing import Dict


class BaseDataset:
    def __init__(self, 
        parent_directory: str, 
        extension: str) -> None:
        """Base dataset to inherit from. Support for reading all files of a specific extension from a parent directory
        
        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib
        extension
            The extension we want to include in our search from the parent_directory directory

        Returns
        -------
        None	

        Usage
        ----------
        parent_directory = 'data'
        extension = "jpg"

        dataset = BaseDataset(parent_directory, extension)
        """
        self.parent_directory = Path(parent_directory)
        self.extension = extension
		
    def read_files(self) -> Iterable:
        """ Construct iterable that extracts file path

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
        """ Show files inside the iterable

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


    def create_dataset_array(
        self, 
        labels: List,
        verbose: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        """Create full dataset array. The columns are relative path to file and its label

        Parameters
        ----------
        labels

        Usage
        ----------
        labels = {"111": male, "112": female}

        Returns
        -------
        namedtuple of X and y	
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
            for label in labels:
                if re.search(label, str(relative_path_without_name)):
                    filename.append(str(relative_path_with_name))
                    target.append(label)

        if verbose:
            print("Finished creating whole dataset array")

        dataset = Dataset(np.array(filename), np.array(target))
        return dataset

    @staticmethod
    def convert_label_array(mapping_of_current_label_and_desired_label, labels):
        """Convert current 

        Parameters
        ----------
        labels

        Usage
        ----------
        mapping_of_current_label_and_desired_label = {"male": 0, "female": 1}
        convert_label_array(mapping_of_current_label_and_desired_label, )

        Returns
        -------
        Tuple of X and y	
        """
        for current_label, desired_label in mapping_of_current_label_and_desired_label.items():
            labels[current_label == labels] = desired_label
        return labels



