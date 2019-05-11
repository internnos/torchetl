# import glob
# import shutil
# import os
from pathlib import Path
import re
from typing import Dict, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

real_path = Path.cwd() / 'data' / 'Replay-Attack' / 'ORI' / 'real'
attack_path = Path.cwd() / 'data' / 'Replay-Attack' / 'ORI' / 'attack'
ori_path = Path.cwd() / 'data' / 'Replay-Attack' / 'ORI' 


class CreatePytorchDatasetFormat:
	
	def __init__(self, origin: str):
		self.origin = origin
		
	def _read_origin(self):
		if self.origin.exists():
			p = Path(self.origin)
			for path in p.rglob("*.jpg"):
				yield path
		else:
			raise ValueError("Directory does not exist")

	def show_origin(self):
		"""
		simulate the images path you're inputting
		"""
		for image_path in self._read_origin():
			print(image_path)

	def simulate_mark_and_recall(self, levels: int):
		"""
		simulate the destination path of your image
		"""
		for destination_path, new_name in self.mark(levels):
			print("Mark:", destination_path / new_name)

	def mark(self, levels: int):
		"""
		Mark destination
		"""
		for image_path in self._read_origin():
			destination_path = image_path.parents[levels-1].resolve()
			parts = image_path.parts[-levels:]
			new_name = ''.join(str(part) for part in parts)
			yield destination_path, new_name

	def recall(self, levels: int, verbose=False):
		"""
		levels correspond to how many levels you want to move up
		"""
		for image_path, (destination_path, new_name) in zip(self._read_origin(), self.mark(levels)):
			image_path.replace(destination_path / new_name)
			if verbose:
				print(image_path, "is moved to", destination_path / new_name)


class PartitionPytorchDataset(CreatePytorchDatasetFormat):
	"""
	Partition Pytorch dataset format to train, validation, and test. Since it inherits 
	"""
	def __init__(self, dataset_path: str, labels: List, training_size: float, test_size:float, random_state: int):
		super().__init__
		self.origin = dataset_path
		self.labels = labels
		self.training_size = training_size
		self.test_size = test_size
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
		X, Y = self._create_dataset_array()
		sss = StratifiedShuffleSplit(n_splits=1, train_size=self.training_size, test_size=self.test_size, random_state=self.random_state)

		for trainIndex, validationTestIndex in sss.split(X, Y):
			xTrain, xValidationTest = X[trainIndex], X[validationTestIndex]
			yTrain, yValidationTest = Y[trainIndex], Y[validationTestIndex]

		sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5,random_state=self.random_state)
		for validationIndex, testIndex in sss.split(xValidationTest, yValidationTest):
			xValidation, xTest = xValidationTest[validationIndex], xValidationTest[testIndex]
			yValidation, yTest = yValidationTest[validationIndex], yValidationTest[testIndex]
		# breakpoint()

		return xTrain,yTrain,xValidation,yValidation,xTest,yTest

	def mark(self, levels):
		"""
		private method to generate destination path. 
		Levels correspond to how much you want to move up in 
		"""
		x_train, y_train, x_validation, y_validation, x_test, y_test = self._stratify_sampling()
		
		for destination_path in x_train:
			for label in self._encode_labels():	
				if bool(re.search(label, destination_path)):
					yield (Path(destination_path), Path(destination_path).parents[levels] / 'train' / label / Path(destination_path).name)

		for destination_path in x_validation:
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
		for destination_path in self.mark(levels):
			print("mark: "destination_path[0], /n "recall;", destination_path[1])
		
	def recall(self, levels: int):
		for destination_path in self.mark(levels):
			origin = destination_path[0]
			destination = destination_path[1]
			# breakpoint()
			Path(destination).parents[0].mkdir(parents=True, exist_ok=True)
			destination_path[0].replace(destination_path[1])




