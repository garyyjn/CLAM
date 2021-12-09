import numpy as np
import os
data_path = 'home/ext_yao_gary_mayo_edu/FuseMount/datasets/TCGA_BLCA'

for filename in os.listdir(data_path):
    print(os.path.join(data_path, filename))