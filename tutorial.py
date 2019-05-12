from mark_and_recall import CreatePytorchDatasetFormat, PartitionPytorchDatasetFormat
from pathlib import Path

data = Path.cwd() / 'data' / 'ori' / 'real' 



a = CreatePytorchDatasetFormat(origin=data, extension="jpg")
#a.show_origin()
a.simulate_mark_and_recall(levels=2)
#a.mark(levels=1)