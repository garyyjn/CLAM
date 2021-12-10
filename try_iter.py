import numpy as np
import os
from feature_extraction import small_feature_extraction
from tqdm import tqdm
import torch
data_path = '/Volumes/HwangSSD2/TCGA_BLCA'
output_path = '/Users/M261759/Documents/GitHub/CLAM/example_bladder_output'
from models.resnet_custom import ResNet_Baseline, resnet50_baseline

feature_extractor = resnet50_baseline(pretrained=True)
if torch.cuda.is_available():
    feature_extractor = feature_extractor.cuda()
for filename in tqdm(os.listdir(data_path)):
    if filename.endswith('.svs'):
        full_path = os.path.join(data_path, filename)
        small_feature_extraction(filename, full_path, output_path, feature_extractor=feature_extractor)
