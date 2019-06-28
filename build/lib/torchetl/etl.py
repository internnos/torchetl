    # standardlib
import csv
import re
from pathlib import Path, PosixPath
from typing import List, Tuple, Callable


# external package
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np

# user
from torchetl.base.dataset import BaseDataset

import pdb

class ExtractThreePartitions(BaseDataset):
    def __init__(self, 
                parent_directory: PosixPath, 
                extension: str,
                labels: List[str], 
                train_size: float, 
                random_state: int, 
                verbose: bool) -> None:
        """Class for creating csv files of train, validation, and test

        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib. Use full path
            instead of relative path
        extension
            The extension we want to include in our search from the parent_directory
        labels

        Returns
        -------
        None	
        """
        super().__init__(parent_directory, extension, labels, train_size, random_state, verbose)

    def _stratify_sampling(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply stratify sampling. Recommended for classification tasks

        Parameters
        ----------
        None

        Returns
        -------
        Tuple of train(X, y), validation(X, y), and test(X, y)	
        """

        x, y = self.create_dataset_array()

        sss = StratifiedShuffleSplit(n_splits=1, train_size=self.train_size, test_size=self.test_size, random_state=self.random_state)

        for train_index, validation_test_index in sss.split(x, y):
            x_train, x_validation_test = x[train_index], x[validation_test_index]
            y_train, y_validation_test = y[train_index], y[validation_test_index]

        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5,random_state=self.random_state)
        for validation_index, test_index in sss.split(x_validation_test, y_validation_test):
            x_validation, x_test = x_validation_test[validation_index], x_validation_test[test_index]
            y_validation, y_test = y_validation_test[validation_index], y_validation_test[test_index]

        train = np.c_[x_train, y_train]
        validation = np.c_[x_validation, y_validation]
        test = np.c_[x_test, y_test]

        if self.verbose:
            print("Finished splitting dataset into train, validation, and test")
        return train, validation, test

    def _random_sampling(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = self.create_dataset_array()
        x_train, x_validation_test, y_train, y_validation_test = train_test_split(x, 
                                                                                  y, 
                                                                                  train_size = self.train_size,
                                                                                  test_size = self.test_size, 
                                                                                  random_state=self.random_state)

        x_validation, x_test, y_validation, y_test = train_test_split(x_validation_test,
                                                                      y_validation_test,
                                                                      train_size = 0.5,
                                                                      test_size = 0.5,
                                                                      random_state = self.random_state)

        train = np.c_[x_train, y_train]
        validation = np.c_[x_validation, y_validation]
        test = np.c_[x_test, y_test]

        if self.verbose:
            print("Finished splitting dataset into train, validation, and test")
        return train, validation, test


    def extract(self, file_prefix: str, save_path: str, is_random_sampling: bool):
        """Create csv file of train, validation, and test

        Parameters
        ----------
        file_prefix
            The prefix of train, validation, and test file_prefix
            Have the format of file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        save_path
            The parent_directory folder name of file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        is_random_sampling
            extract train, validation, and test based on random sampling. If set to false, then stratify sampling is applied.
            The best practice is to use stratify sampling for classification tasks and random sampling for regression tasks
        Returns
        -------
        train, validation, and test csv with the following name:
        file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv	
        """
        if is_random_sampling:
            train, validation, test = self._random_sampling()
        else:
            train, validation, test = self._stratify_sampling()

        save_into = Path.cwd() / save_path
        save_into.mkdir(parents=True, exist_ok=True)

        with open(f'{save_path}/{file_prefix}_train.csv', 'w') as writer:
            csv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)

        if self.verbose:
            print(f'Finished writing {file_prefix}_train.csv into {save_into}')
        

        with open(f'{save_path}/{file_prefix}_validation.csv', 'w') as writer:
            csv_writer = csv.writer(writer)
            for row in validation:
                csv_writer.writerow(row)

        if self.verbose:
            print(f'Finished writing {file_prefix}_validation.csv into {save_into}')

        with open(f'{save_path}/{file_prefix}_test.csv', 'w') as writer:
            csv_writer = csv.writer(writer)
            for row in test:
                csv_writer.writerow(row)

        if self.verbose:
            print(f'Finished writing {file_prefix}_test.csv into {save_into}')


class ExtractTwoPartitions(BaseDataset):
    def __init__(self, 
                parent_directory: PosixPath, 
                extension: str,
                labels: List[str], 
                train_size: float, 
                random_state: int, 
                verbose: bool) -> None:
        """Class for creating csv files of train, validation

        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib. Use full path
            instead of relative path
        extension
            The extension we want to include in our search from the parent_directory
        labels

        Returns
        -------
        None	
        """
        super().__init__(parent_directory, extension, labels, train_size, random_state, verbose)

    def _stratify_sampling(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply stratify sampling. Recommended for classification tasks

        Parameters
        ----------
        None

        Returns
        -------
        Tuple of train(X, y), validation(X, y), and test(X, y)	
        """

        x, y = self.create_dataset_array()

        sss = StratifiedShuffleSplit(n_splits=1, train_size=self.train_size, test_size=self.test_size, random_state=self.random_state)

        for train_index, validation_test_index in sss.split(x, y):
            x_train, x_validation_test = x[train_index], x[validation_test_index]
            y_train, y_validation_test = y[train_index], y[validation_test_index]


        train = np.c_[x_train, y_train]
        validation = np.c_[x_validation_test, y_validation_test]


        if self.verbose:
            print("Finished splitting dataset into train, validation")
        return train, validation

    def _random_sampling(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = self.create_dataset_array()
        x_train, x_validation_test, y_train, y_validation_test = train_test_split(x, 
                                                                                  y, 
                                                                                  train_size = self.train_size,
                                                                                  test_size = self.test_size, 
                                                                                  random_state=self.random_state)


        train = np.c_[x_train, y_train]
        validation = np.c_[x_validation_test, y_validation_test]

        if self.verbose:
            print("Finished splitting dataset into train, validation, and test")
        return train, validation


    def extract(self, file_prefix: str, save_path: str, is_random_sampling: bool):
        """Create csv file of train, validation

        Parameters
        ----------
        file_prefix
            The prefix of train, validation, and test file_prefix
            Have the format of file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        save_path
            The parent_directory folder name of file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv
        is_random_sampling
            extract train, validation, and test based on random sampling. If set to false, then stratify sampling is applied.
            The best practice is to use stratify sampling for classification tasks and random sampling for regression tasks
        Returns
        -------
        train, validation, and test csv with the following name:
        file_prefix_train.csv, file_prefix_validation.csv, and test_validation.csv	
        """
        if is_random_sampling:
            train, validation = self._random_sampling()
        else:
            train, validation = self._stratify_sampling()

        save_into = Path.cwd() / save_path
        save_into.mkdir(parents=True, exist_ok=True)

        with open(f'{save_path}/{file_prefix}_train.csv', 'w') as writer:
            csv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)

        if self.verbose:
            print(f'Finished writing {file_prefix}_train.csv into {save_into}')
        

        with open(f'{save_path}/{file_prefix}_validation.csv', 'w') as writer:
            csv_writer = csv.writer(writer)
            for row in validation:
                csv_writer.writerow(row)

        if self.verbose:
            print(f'Finished writing {file_prefix}_validation.csv into {save_into}')



class TransformAndLoad(Dataset):
    def __init__(self, 
                parent_directory: str, 
                extension: str, 
                csv_file: str, 
                transform: Callable = None,
                is_bbox_available : bool = False) -> None:
        """Class for reading csv files of train, validation, and test

        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib
        extension
            The extension we want to include in our search from the parent_directory directory
        csv_file
            The path to csv file containing X and y
        Transform
            Callable which apply transformations

        Returns
        -------
        None	
        """
        self.parent_directory = Path(parent_directory)
        self.extension = extension
        self.transform = transform
        self.is_bbox_available = is_bbox_available
        try:
            self.csv_file = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f'{Path.cwd() / csv_file} does not exist')

    def __len__(self) -> int:
        """Return the length of the dataset

        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib
        extension
            The extension we want to include in our search from the parent_directory directory
        csv_file
            The path to csv file containing X and y
        Transform
            Callable which apply transformations

        Returns
        -------
        Length of the dataset	
        """
        return len(self.csv_file)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the X and y of a specific instance based on the index

        Parameters
        ----------
        idx
            The index of the instance 

        Returns
        -------
        Tuple of X and y of a specific instance	
        """
        image_path = self.parent_directory / self.csv_file.iloc[idx, 0]
        target = self.csv_file.iloc[idx, 1]
        image_array = cv2.imread(str(image_path))
        
        if self.is_bbox_available:
            x_min, y_min, x_max, y_max = self.csv_file.iloc[idx, 2:]
            image_array = image_array[y_min:y_max, x_min:x_max]
        if self.transform:
            image_array = self.transform(image_array)

        return image_array, target