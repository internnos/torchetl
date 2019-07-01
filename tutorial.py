import cv2
import numpy as np
# from nodeflux.face_detection.retinaface import RetinaFace
import time
import pdb
# from nodeflux.face_spoof.preprocessing.transform import center_crop
# from nodeflux.face_spoof.feathernet import FeatherNetA
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import transforms
from torchetl.etl import TransformAndLoad
import pdb
import pandas as pd
import pandas as pd

# pdb.set_trace()
parent_directory = Path('data') / 'imdb'

test_dataset_csv = parent_directory / 'imdb_even_cleaner_with_bbox.csv'


df = pd.read_csv(test_dataset_csv)
# pdb.set_trace()
testing_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])

test_dataset = TransformAndLoad(parent_directory=Path(parent_directory), 
                                extension="jpg", 
                                csv_file=test_dataset_csv, 
                                # transform=testing_pipeline,
                                is_bbox_available=True,
                                resize_to=(640,480)
                                )

for index, value in enumerate(test_dataset):
    image, label = test_dataset.__getitem__(index)[0], test_dataset.__getitem__(index)[1]
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()