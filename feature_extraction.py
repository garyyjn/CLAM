import numpy as np
import cv2
import openslide
from tqdm import tqdm
import torch
import os
import pickle

from models.resnet_custom import ResNet_Baseline, resnet50_baseline
import openslide
feature_extractor = resnet50_baseline(pretrained=True)
data_path = "/Volumes/HwangSSD2/TCGA_BLCA"
output_path = "/Users/M261759/Documents/GitHub/CLAM/example_bladder_output"#class1/features, class1/dictionary, class2/features, class2/dicitonaru
annotation_path = ""
print("fuck")
def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    num_pixels = patch.size
    return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    num_pixels = patch.size
    return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def small_feature_extraction(slide_name, file_path, output_path, feature_extractor, tile_dims = [224,224], check_filter = True,):
    slide = openslide.OpenSlide(file_path)
    index_xy = {}
    slide_x, slide_y = slide.dimensions
    print("Working on: {}".format(slide_name))
    print("Whole Slide Dims: {}".format(slide.dimensions))
    tile_dim_x, tile_dim_y = tile_dims
    num_tiles_x, num_tiles_y =  (int)(slide_x/tile_dim_x), (int)(slide_y/tile_dim_y)
    total_tiles = (int)(slide_x/tile_dim_x)*(int)(slide_y/tile_dim_y)
    print("Number of tiles: {}".format(total_tiles))
    post_filter_tiles = 0
    whole_slide_in_mem = np.array(slide.read_region(location=(0,0), level = 1, size = (slide_x, slide_y)))
    for i_x in tqdm(range(num_tiles_x)):
        for i_y in range(num_tiles_y):
            curr_x = i_x * tile_dim_x
            curr_y = i_y * tile_dim_y
            tile_curr = whole_slide_in_mem[curr_x:curr_x+tile_dim_x, curr_y:curr_y+ tile_dim_y,0:3]
            if check_filter:
                if isBlackPatch_S(tile_curr, rgbThresh=20, percentage=0.05) or isWhitePatch_S(tile_curr, rgbThresh=220,
                                                                                              percentage=0.25):
                    continue

            index_xy.update({post_filter_tiles: (curr_x, curr_y)})
            post_filter_tiles += 1
    print("Post filter tiles: {}".format(post_filter_tiles))

    #print(index_xy)
    output = np.zeros(shape=(post_filter_tiles, 1024))
    for i in range(post_filter_tiles):
        curr_x, curr_y = index_xy[i]
        tile_curr = slide.read_region(location=(curr_x, curr_y), level=1, size=(tile_dim_x, tile_dim_y))
        tile_curr = np.array(tile_curr)[:, :, 0:3]
        tile_curr = torch.unsqueeze(torch.tensor(np.transpose(np.array(tile_curr),(2,0,1))),dim=0)/255
        if torch.cuda.is_available():
            tile_curr = tile_curr.cuda()
        features = feature_extractor(tile_curr)
        output[i, :] = features.detach().numpy()
    with open(os.path.join(output_path,'features',"{}.npy".format(slide_name)),'wb') as f:
        np.save(f, output)
    f = open(os.path.join(output_path,'dictionaries',"{}.dict".format(slide_name)), "wb")
    pickle.dump(index_xy, f)
    f.close()
    return output, index_xy
