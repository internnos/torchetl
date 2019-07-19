    # standardlib
import csv
import re
from pathlib import Path, PosixPath
from typing import List, Tuple, Callable, Optional
from collections import namedtuple

# external package
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from skimage import transform as trans

# user
from torchetl.base.dataset import BaseDataset

import pdb

class Extract():
    def __init__(self):
        pass

    @staticmethod
    def stratify(x, y, train_size, random_state=69):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=1-train_size, random_state=random_state)
        Partition = namedtuple('Partition', ['x_train', 'y_train', 'x_test', 'y_test'])
        for train_index, test_index in sss.split(x, y):
            partition = Partition
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

        partition = Partition(x_train, y_train, x_test, y_test)
        return partition

    @staticmethod
    def random(x, y, train_size, random_state=69):
        Partition = namedtuple('Partition', ['x_train', 'y_train', 'x_test', 'y_test'])
        x_train, x_test, y_train, y_test = train_test_split(
                                                            x, 
                                                            y, 
                                                            train_size = train_size,
                                                            test_size = 1 - train_size, 
                                                            random_state = random_state)

        partition = Partition(x_train, y_train, x_test, y_test)
        return partition

    @staticmethod
    def dump_to_csv(x, y, dump_to):
        df = pd.DataFrame(data=x, columns=y)
        df.to_csv(dump_to, index=False)


class TransformAndLoad():
    def __init__(self, 
                parent_directory: str, 
                csv_file: str, 
                bounding_box_column_index: List[int],
                landmark_column_index: List[int], 
                transform: Callable = None,
                apply_face_cropping : bool = False,
                apply_face_alignment: bool = False,
                ) -> None:
        """Class for reading csv files of train, validation, and test

        Parameters
        ----------
        parent_directory
            The parent_directory folder path. It is highly recommended to use Pathlib
        extension
            The extension we want to include in our search from the parent_directory directory
        csv_file
            The path to csv file containing the information of the image. For the bare minimum it should contain path and label.
            If apply_face_cropping is set to True, then it must contain bounding box for index 2 until 5
            If apply_face_alignment is set to True, then it must contain bounding box for index 6 until 15
        transform
            Callable which apply transformations
        apply_face_cropping
            Read description in csv_file
            In addition, if apply_face_cropping is set to True, then apply_face_alignment must be set to False
        resize_to
            Resize input image to this value. Always set this value. By default is valued at (640,480)
        apply_face_alignment
            Read description in csv_file
            In addition, if apply_face_alignment is set to True, then apply_face_cropping must be set to False


        Returns
        -------
        None	
        """
        self.parent_directory = Path(parent_directory)
        self.transform = transform
        self.apply_face_cropping = apply_face_cropping
        self.apply_face_alignment = apply_face_alignment
        self.bounding_box_column_index = bounding_box_column_index
        self.landmark_column_index = landmark_column_index

        try:
            self.csv_file = pd.read_csv(str(csv_file))
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
        csv_filecsv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)
            The path to csv file containcsv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)ing X and y
        Transformcsv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)
            Callable which apply transfocsv_writer = csv.writer(writer)
            for row in train:
                csv_writer.writerow(row)rmations

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
        # pdb.set_trace()
        image_path = self.parent_directory / self.csv_file.iloc[idx, 0]
        target = self.csv_file.iloc[idx, 1]
        image_array = cv2.imread(str(image_path))
        
        if self.apply_face_cropping and self.bounding_box_column_index:
            assert not self.apply_face_alignment
    
            bounding_box_index_start, bounding_box_index_end = self.bounding_box_column_index
            x_min, y_min, x_max, y_max = self.csv_file.iloc[idx, bounding_box_index_start:bounding_box_index_end+1].astype(int)
            image_array = image_array[y_min:y_max, x_min:x_max]

        if self.apply_face_alignment and self.landmark_column_index:
            assert not self.apply_face_cropping

            src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
            
            landmark_index_start, landmark_index_end = self.landmark_column_index
            landmark = self.csv_file.iloc[idx, landmark_index_start:landmark_index_end+1]
            landmark = np.array(landmark, ndmin=2)
            landmark = landmark.reshape(5,2)
            dst = landmark.astype(np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2, :]
            image_array = cv2.warpAffine(image_array, M, (112, 112), borderValue=0.0)

        if self.transform:
            image_array = self.transform(image_array)

        return image_array, target