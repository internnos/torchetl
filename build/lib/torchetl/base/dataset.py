from pathlib import Path, PosixPath
import pdb
from typing import Iterable, Sequence, List, Tuple
import re
import numpy as np

class BaseDataset:
	def __init__(self, 
        parent_directory: PosixPath, 
        extension: str,
        labels: List[str],
        train_size: float,
        random_state: int,
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
		self.labels = labels
		self.train_size = train_size
		self.test_size = 1 - train_size
		self.random_state = random_state
		self.verbose = verbose
	
	def read_parent_directory(self) -> str:
		""" Return parent_directory directory

		Parameters
		----------
		None

		Returns
		-------
		None	
		"""
		return self.parent_directory
		
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
		print (files[:n])

	def create_dataset_array(self) -> Tuple[np.ndarray, np.ndarray]:
		"""Create full dataset array from reading files

		Parameters
		----------
		None

		Returns
		-------
		Tuple of X and y	
		"""
		target = []
		filename = []
		for parent_directory in self.read_files():
			# list of folders leading to filename
			child = parent_directory.parent.parts[len(Path.cwd().parts)+1:]
			# path of folders leading to filename
			child = Path(*child)
			for encoded_label, label in enumerate(self.labels):
				# not including the filename, only the directories leading to it
				if re.search(label, str(child)):
					# append child path
					filename.append(str(child / parent_directory.name))
					target.append(encoded_label)
		if self.verbose:
			print("Finished creating whole dataset array")

		return np.array(filename), np.array(target)