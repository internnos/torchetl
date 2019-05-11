# import glob
# import shutil
# import os
from pathlib import Path
import re
from typing import Dict, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


# def load_images_from_directory(source_folder):
#   os.chdir(source_folder)
#   imageRelativePath = glob.glob("**/*.jpg", recursive=True)
#   return imageRelativePath

# def open_txt_file(txt_file):
#   with open(txt_file, 'r') as images:
#     for imageRelativePath in images:
#         imageRelativePath = imageRelativePath.rstrip('\n')
#         yield imageRelativePath

# def copy_images_from_txt(txt_file, source_folder, destination_folder):
#   Path(destination_folder).mkdir(parents=True, exist_ok=True)
#   for sourceImageRelativePath in open_txt_file(txt_file): 
#     try:
#       sourceImageFullPath = os.path.join(source_folder, sourceImageRelativePath)
#       shutil.copy(sourceImageFullPath, destination_folder)
#     except:
#       continue

# def copy_images_to_obey_pytorch_dataset_loader(source_folder, labels, copy=True):
#   os.chdir(source_folder)
#   imageRelativePath = glob.glob("**/*.jpg", recursive=True)

#   for label in labels:
#     destinationFolder = os.path.join

#   for sourceImageRelativePath in imageRelativePath:
#     sourceImageFullPath = os.path.join(source_folder, sourceImageRelativePath)
#     for label in labels:
#       if bool(re.search(label,sourceImageRelativePath)):
#         destinationFolder = os.path.join(source_folder, label)
#     try:
#         if copy:
#           shutil.copy(sourceImageFullPath, destinationFolder)
#         else:
#           shutil.move(sourceImageFullPath, destinationFolder)
#     except FileNotFoundError:
#       pass





# def partition_images_into_train_val_test(images_list, source_folder, destination_folder, labels, copy=True):
#   for label in labels:
#     Path(os.path.join(destination_folder,label)).mkdir(parents=True, exist_ok=True)
#   for sourceImageRelativePath in images_list:
#     sourceImageFullPath = os.path.join(source_folder, sourceImageRelativePath)
#     try:
#         if copy:
#           shutil.copy(sourceImageFullPath, os.path.join(destination_folder,sourceImageRelativePath))
#         else:
#           shutil.move(sourceImageFullPath, os.path.join(destination_folder,sourceImageRelativePath))
#     except FileNotFoundError:
#       pass


# def encode_labels(labels):
#   labels_dictionary = {}
#   for (index,label) in enumerate(labels):
#     labels_dictionary[label]=index
#   return labels_dictionary 


# def _create_dataset_array(parent_folder, labels_dictionary):
#   import numpy as np

#   imageRelativePathArray = []
#   targetArray = []
#   imagesList = load_images_from_directory(parent_folder)
#   for imageRelativePath in imagesList:
#     for label in labels_dictionary:
#       if bool(re.search(label,imageRelativePath)):
#         imageRelativePathArray.append(imageRelativePath)
#         targetArray.append(labels_dictionary[label])
#   return (np.array(imageRelativePathArray), np.array(targetArray))

# def common_member(a, b): 
#   a_set = set(a) 
#   b_set = set(b) 
#   if (a_set & b_set): 
#       return True 
#   else: 
#       return False

def stratify_sampling(parent_folder, labels_dictionary, training_size=0.7, test_size=0.3,is_dump=True, random_state=69):
  from sklearn.model_selection import StratifiedShuffleSplit
  import numpy as np

  X, Y = _create_dataset_array(parent_folder, labels_dictionary)
  sss = StratifiedShuffleSplit(n_splits=1, train_size=training_size, test_size=test_size, random_state=random_state)

  for trainIndex, validationTestIndex in sss.split(X, Y):
    xTrain, xValidationTest = X[trainIndex], X[validationTestIndex]
    yTrain, yValidationTest = Y[trainIndex], Y[validationTestIndex]

  sss = StratifiedShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5,random_state=random_state)
  for validationIndex, testIndex in sss.split(xValidationTest, yValidationTest):
    xValidation, xTest = xValidationTest[validationIndex], xValidationTest[testIndex]
    yValidation, yTest = yValidationTest[validationIndex], yValidationTest[testIndex]

  return xTrain,yTrain,xValidation,yValidation,xTest,yTest


# labels=encode_labels(['real','attack'])


# # mfsd Ori
# mfsdOriParentFolder =  "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/ORI/"
# mfsdOriAttackFolder =  "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/ORI/attack/"
# mfsdOriRealFolder = "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/ORI/real/"
# mfsdOriTrainingFolder = "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/ORI/training/"
# mfsdOriValidationFolder = "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/ORI/validation/"
# mfsdOriTestFolder = "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/ORI/test/"

# copy_images_to_obey_pytorch_dataset_loader(mfsdOriParentFolder, labels)
# #copy_images_to_obey_pytorch_dataset_loader(mfsdOriParentFolder, mfsdOriParentFolder, mfsdOriParentFolder)

# mfsdOriAttackImageList = load_images_from_directory(mfsdOriAttackFolder)
# mfsdOriRealImageList = load_images_from_directory(mfsdOriRealFolder)


# xTrain,yTrain,xValidation,yValidation,xTest,yTest = stratify_sampling(mfsdOriParentFolder, labels)
# partition_images_into_train_val_test(xTrain, mfsdOriParentFolder, mfsdOriTrainingFolder, labels)
# partition_images_into_train_val_test(xValidation, mfsdOriParentFolder, mfsdOriValidationFolder, labels)
# partition_images_into_train_val_test(xTest, mfsdOriParentFolder, mfsdOriTestFolder, labels)

# # mfsd Photo
# mfsdPhotoParentFolder =  "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/mfsdPhoto/"
# mfsdPhotoAttackFolder =  "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/mfsdPhoto/attack/"
# mfsdPhotoRealFolder = "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/mfsdPhoto/real/"
# mfsdPhotoTrainingFolder = "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/mfsdPhoto/training/"
# mfsdPhotoValidationFolder = "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/mfsdPhoto/validation/"
# mfsdPhotoTestFolder = "/Users/jedi/Repo/Nodeflux/Dataset/MFSD/mfsdPhoto/test/"

# mfsdPhotoAttackImageList = load_images_from_directory(mfsdPhotoAttackFolder)
# mfsdPhotoRealImageList = load_images_from_directory(mfsdPhotoRealFolder)

# labels=encode_labels(['real','attack'])
# xTrain,yTrain,xValidation,yValidation,xTest,yTest = stratify_sampling(mfsdPhotoParentFolder, labels)
# partition_images_into_train_val_test(xTrain, mfsdPhotoParentFolder, mfsdPhotoTrainingFolder, labels)
# partition_images_into_train_val_test(xValidation, mfsdPhotoParentFolder, mfsdPhotoValidationFolder, labels)
# partition_images_into_train_val_test(xTest, mfsdPhotoParentFolder, mfsdPhotoTestFolder, labels)

real_path = Path.cwd() / 'data' / 'Replay-Attack' / 'ORI' / 'real'
attack_path = Path.cwd() / 'data' / 'Replay-Attack' / 'ORI' / 'attack'
ori_path = Path.cwd() / 'data' / 'Replay-Attack' / 'ORI' 


class CreatePytorchDatasetFormat:
	def __init__(self, dataset_path: str):
		self.dataset_path = dataset_path
	def _read_images(self):
		if self.dataset_path.exists():
			p = Path(self.dataset_path)
			for path in p.rglob("*.jpg"):
				yield path
		else:
			raise ValueError("Directory does not exist")

	def _create_destination_path(self, levels: int):
		"""
		Private method for destination_path and rename_and_move_images
		"""
		for image_path in self._read_images():
			destination_path = image_path.parents[levels-1].resolve()
			parts = image_path.parts[-levels:]
			new_name = ''.join(str(part) for part in parts)
			yield destination_path, new_name

	def show_source_path(self):
		"""
		simulate the images path you're inputting
		"""
		for image_path in self._read_images():
			print(image_path)

	def simulate_relocation(self, levels: int):
		"""
		simulate the destination path of your image
		"""
		for destination_path, new_name in self._create_destination_path(levels):
			print(destination_path / new_name)

	def relocate_images(self, levels: int, verbose=False):
		"""
		levels correspond to how many levels you want to move up
		"""
		for image_path, (destination_path, new_name) in zip(self._read_images(), self._create_destination_path(levels)):
			image_path.replace(destination_path / new_name)
			if verbose:
				print(image_path, "is moved to", destination_path / new_name)


class PartitionPytorchDataset(CreatePytorchDatasetFormat):
	"""
	Partition Pytorch dataset format to train, validation, and test. Since it inherits 
	"""
	def __init__(self, dataset_path: str, labels: List, training_size: float, test_size:float, random_state: int):
		super().__init__
		self.dataset_path = dataset_path
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
		for image_path in self._read_images():
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

	def _create_destination_path(self, levels):
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


	def simulate_relocation(self, levels: int):
		"""
		run this to simulate the relocation.
		level denotes how much you want to move up in the directory 
		"""
		for destination_path in self._create_destination_path(levels):
			print(destination_path[0], "would be moved to", destination_path[1])
		
	def relocate(self, levels: int):
		for destination_path in self._create_destination_path(levels):
			origin = destination_path[0]
			destination = destination_path[1]
			# breakpoint()
			Path(destination).parents[0].mkdir(parents=True, exist_ok=True)
			destination_path[0].replace(destination_path[1])


		

			






# a = CreatePytorchDatasetFormat(attack_path)
# a.show_source_path()
#a.simulate_destination_path(3)
#a.rename_and_move_images(3)

b = PartitionPytorchDataset(ori_path, ["attack", "real"], 0.7, 0.3, 69)
# b.simulate_relocation(1)
b.relocate(1)
# a,b,c,d,e,f = b._stratify_sampling()
# print(e)
# labels = b.encode_labels(["attack", "real"])

# #print(labels)

# #b.show_source_path()
# b.simulate_destination_path(labels, 0.7, 0.3, 69, 1)




