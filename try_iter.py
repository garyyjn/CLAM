import numpy as np
import os
from feature_extraction import small_feature_extraction
from tqdm import tqdm
data_path = '/home/ext_yao_gary_mayo_edu/FuseMount/datasets/TCGA_BLCA'
output_path = '/home/ext_yao_gary_mayo_edu/FuseMount/post_process/TCGA_BLCA_ResNet50'
from models.resnet_custom import ResNet_Baseline, resnet50_baseline

feature_extractor = resnet50_baseline(pretrained=True).cuda()
for filename in tqdm(os.listdir(data_path)):
    if filename.endswith('.svs'):
        full_path = os.path.join(data_path, filename)
        small_feature_extraction(filename, full_path, output_path, feature_extractor=feature_extractor)
