import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
from models.model_clam import CLAM_SB, CLAM_MB


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#create full CLAM pipline


#load the two models
#patches -> features

print('loading model checkpoint')
model = resnet50_baseline(pretrained=True)
model = model.to(device)

#extract top k important

CLAM = CLAM_MB()