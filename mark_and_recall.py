# import glob
# import shutil
# import os
from pathlib import Path
import re
from typing import Dict, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class CreatePytorchDatasetFormat:
	"""

	"""
	def __init__(self, origin: str, extension:str):
		self.origin = origin
		self.extension = extension
		
	def _read_origin(self):
		"""
		private method to only read files
		"""
		if self.origin.exists():
			p = Path(self.origin)
			# breakpoint()
			for path in p.rglob("*." + self.extension):
				yield path
		else:
			raise ValueError("Directory does not exist")

	def show_origin(self):
		"""
		show the origin, directly related with _read_origin
		"""
		for image_path in self._read_origin():
			print(image_path)

	def simulate_mark_and_recall(self, levels: int):
		"""
		simulate the mark and recall process
		levels correspond to how many levels you want to move up
		"""
		for origin, destination in zip(self._read_origin(), self._mark(levels)):
			print("Origin:", origin, " Destination:", destination)

	def _mark(self, levels: int):
		"""
		Private method Mark desired destination
		"""
		for origin in self._read_origin():
			destination = origin.parents[levels-1]
			parts = origin.parts[-levels:]
			new_name = ''.join(str(part) for part in parts)
			destination = destination / new_name
			yield destination

	def mark_and_recall(self, levels: int):
		"""
		levels correspond to how many levels you want to move up
		"""
		for origin, destination in zip(self._read_origin(), self._mark(levels)):
			origin.replace(destination)


class PartitionPytorchDatasetFormat(CreatePytorchDatasetFormat):
	"""
	Partition Pytorch dataset format to train, validation, and test
	"""
	def __init__(self, dataset_path: str, labels: List, training_size: float, random_state: int):
		"""
		"""
		super().__init__
		self.origin = dataset_path
		self.labels = labels
		self.training_size = training_size
		self.test_size = 1-training_size
		self.random_state = random_state

	def _encode_labels(self):
		"""
		private method to encode labesl: List into label; Dict
		"""
		labels_dictionary = {}
		for (index,label) in enumerate(self.labels):
			labels_dictionary[label]=index
		return labels_dictionary 

	def _create_dataset_array(self):
		"""
		private method to apply stratify sampling. In sklearn, we must build the whole array
		"""
		target = []
		name = []
		for image_path in self._read_origin():
			for label in self._encode_labels():
				if re.search(label, str(image_path)):
					# breakpoint()
					name.append(str(image_path))
					target.append(label)
		return np.array(name), np.array(target)

	def _stratify_sampling(self):
		"""
		private method to apply stratify sampling on the created array
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
		# breakpoint()

		return x_train,y_train,x_validation,y_validation,x_test,y_test

	def _mark(self, levels):
		"""
		method to generate mark
		Levels correspond to how much you want to move up in 
		"""
		x_train, y_train, x_validation, y_validation, x_test, y_test = self._stratify_sampling()
		
		for origin in x_train:
			for label in self._encode_labels():	
				if bool(re.search(label, origin)):
					yield (Path(origin), Path(origin).parents[levels] / 'train' / label / Path(origin).name)

		for origin in x_validation:
			for label in self._encode_labels():	
				if bool(re.search(label, destination_path)):
					yield (Path(destination_path), Path(destination_path).parents[levels] / 'validation' / label / Path(destination_path).name)


		for destination_path in x_test:
			for label in self._encode_labels():	
				if bool(re.search(label, destination_path)):
					yield (Path(destination_path), Path(destination_path).parents[levels] / 'test' / label / Path(destination_path).name)


	def simulate_mark_and_recall(self, levels: int):
		"""
		run this to simulate the mark and recall.
		level denotes how much you want to move up in the directory 
		"""
		for destination_path in self._mark(levels):
			print("mark: ", destination_path[0]/n, "recall;", destination_path[1])
		
	def mark_and_recall(self, levels: int):
		for destination_path in self._mark(levels):
			origin = destination_path[0]
			destination = destination_path[1]
			Path(destination).parents[0].mkdir(parents=True, exist_ok=True)
			destination_path[0].replace(destination_path[1])




