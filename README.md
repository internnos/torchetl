# TorchETL

If you're working on classification problem, with dataset that is available in their native format (jpg, bmp, etc) and have PyTorch in your arsenal, you'll most likely feel that the **DatasetFolder** or **ImageFolder** is not good enough. So does vanilla **torch.utils.data.Dataset**. This library attempts to bridge that gap to effectively Extract, Transform, and Load your data by extending **torch.utils.data.Dataset**.  

### Main Features

Extract class would partition your dataset into train, validation, and test csv

TransformAndLoad class would Transform and consume your dataset efficiently

### Prerequisites

Tested on Python 3.7.2

Other versions might work given type checking is supported


### Installing

pip3 install -r requirements.txt


### Tutorial

See Tutorial.ipynb

