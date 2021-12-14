import numpy as np
import os
from feature_extraction import small_feature_extraction, simple_extraction, small_feature_extraction_high_mem, small_feature_extraction_center
from tqdm import tqdm
import torch

local = False
if local:
    data_path = '/Users/M261759/Documents/GitHub/CLAM/example_bladder_data'
    output_path = '/Users/M261759/Documents/GitHub/CLAM/example_bladder_output'
else:
    data_path = '/home/ext_yao_gary_mayo_edu/FuseMount/datasets/TCGA_BLCA'
    output_path = './extraction_output'
from models.resnet_custom import ResNet_Baseline, resnet50_baseline

feature_extractor = resnet50_baseline(pretrained=True)

if torch.cuda.is_available():
    feature_extractor = feature_extractor.cuda()
for filename in tqdm(os.listdir(data_path)):
    if filename.endswith('.svs'):
        full_path = os.path.join(data_path, filename)
        small_feature_extraction_center(filename, full_path, output_path, feature_extractor=feature_extractor)
'''
filename = 'datasets_TCGA_BLCA_TCGA-HQ-A5NE-01Z-00-DX1.8046B606-085B-406F-A8D7-92A3E021020A.svs'
full_path = os.path.join(data_path, filename)
simple_extraction(filename, full_path, output_path, feature_extractor)
'''