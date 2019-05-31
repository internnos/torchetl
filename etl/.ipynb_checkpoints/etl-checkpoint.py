# standardlib
import csv
import re
from pathlib import Path
from typing import List, Tuple, Callable

# external package
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import cv2
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset

# user
from etl.base.dataset import BaseDataset

class Extract(BaseDataset):
	def __init__(self, parent_directory: str, 
				 extension: str,
				 labels: List[str], 
				 training_size: float, 
				 random_state: int, 
				 verbose: bool) -> None:
		"""Class for creating csv files of train, validation, and test

		Parameters
		----------
		parent_directory
			The parent_directory folder path. It is highly recommended to use Pathlib
		extension
			The extension we want to include in our search from the parent_directory directory
		labels

		Returns
		-------
		None	
		"""
		super().__init__(parent_directory, extension)
		self.labels = labels
		self.training_size = training_size
		self.test_size = 1 - training_size
		self.random_state = random_state
		self.verbose = verbose

	def _create_dataset_array(self) -> Tuple[np.ndarray, np.ndarray]:
		"""Sklearn stratified sampling uses a whole array so we must build it first

		Parameters
		----------
		None

		Returns
		-------
		Tuple of X and y	
		"""
		target = []
		name = []
		
		for parent_directory in self.read_files():
			child = parent_directory.parts[len(self.parent_directory.parts):]
			child = '/'.join(str(part) for part in child)
			for encoded_label, label in enumerate(self.labels):
				if re.search(label, child):
					name.append(str(child))
					target.append(encoded_label)
		if self.verbose:
			print("Finished creating whole dataset array")

		return np.array(name), np.array(target)

	def _stratify_sampling(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Sklearn stratified sampling uses a whole array so we must build it first

		Parameters
		----------
		None

		Returns
		-------
		Tuple of train(X, y), validation(X, y), and test(X, y)	
		"""
		x, y = self._create_dataset_array()
		sss = StratifiedShuffleSplit(n_splits=1, train_size=self.training_size, test_size=self.test_size, random_state=self.random_state)

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

	def extract(self, filename: str, save_path: str):
		"""Create csv file of train, validation, and test

		Parameters
		----------
		filename
			The prefix of train, validation, and test filename
			Have the format of filename_train.csv, filename_validation.csv, and test_validation.csv
		save_path
			The parent_directory folder name of filename_train.csv, filename_validation.csv, and test_validation.csv

		Returns
		-------
		train, validation, and test csv with the following name:
		filename_train.csv, filename_validation.csv, and test_validation.csv	
		"""
		train, validation, test = self._stratify_sampling()

		save_into = Path.cwd() / save_path
		save_into.mkdir(parents=True, exist_ok=True)

		with open(f'{save_path}/{filename}_train.csv', 'w') as writer:
			csv_writer = csv.writer(writer)
			for row in train:
				csv_writer.writerow(row)

		if self.verbose:
			print(f'Finished writing {filename}_train.csv into {save_into}')
		

		with open(f'{save_path}/{filename}_validation.csv', 'w') as writer:
			csv_writer = csv.writer(writer)
			for row in validation:
				csv_writer.writerow(row)

		if self.verbose:
			print(f'Finished writing {filename}_validation.csv into {save_into}')

		with open(f'{save_path}/{filename}_test.csv', 'w') as writer:
			csv_writer = csv.writer(writer)
			for row in test:
				csv_writer.writerow(row)

		if self.verbose:
			print(f'Finished writing {filename}_test.csv into {save_into}')

class TransformAndLoad(Dataset):
	def __init__(self, parent_directory: str, 
				 extension: str, 
				 csv_file: str, 
				 transform: Callable = None) -> None:
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
		self.parent_directory = parent_directory
		self.extension = extension
		self.csv_file = pd.read_csv(csv_file)
		self.transform = transform
	
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
		parent_directory = self.parent_directory / self.csv_file.iloc[idx, 0]
		target = self.csv_file.iloc[idx, 1]
		parent_directory = cv2.imread(str(parent_directory))

		if self.transform:
			parent_directory = self.transform(parent_directory)

		return parent_directory, target


def main():

	# create training, validation, and test csv
	parent_directory = Path.cwd() / 'data' 
	a = Extract(parent_directory, "jpg", ["attack", "real"], 0.8, 69, verbose=True)
	a.extract(filename="mfsd", save_path="data/mfsd")

	# check individual dataset
	mfsd_train_csv = str(parent_directory / "mfsd" / "mfsd_train.csv")

	data_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(
						mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225])
	])

	c = TransformAndLoad(parent_directory=parent_directory, extension="jpg", csv_file=mfsd_train_csv, transform=data_transform)
	print(c.__getitem__(0))


if __name__ == "__main__":
	main()