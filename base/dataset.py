from pathlib import Path
import pdb
from typing import Iterable, Sequence

class BaseDataset:
	def __init__(self, parent_directory: str, extension: str) -> None:
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
			for origin in p.rglob("*." + self.extension):
				yield origin
		except:
			raise ValueError("Directory does not exist")

	def show_files(self) -> Sequence:
		""" Show parent_directory directory

		Parameters
		----------
		None

		Returns
		-------
		origin
			the path to read files
		"""
		for image_path in self.read_files():
			print(image_path)


def main():
	parent_directory = Path.cwd().parent_directorys[0] / "data" / "ori" / "attack"
	a = BaseDataset(parent_directory = parent_directory, extension="jpg")
	a.show_parent_directory() 

if __name__ == "__main__":
	main()