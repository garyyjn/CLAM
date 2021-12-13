import numpy as np
import cv2
import openslide
from tqdm import tqdm
import torch
import os
import pickle
from shutil import copyfile, copy, copy2
from models.resnet_custom import ResNet_Baseline, resnet50_baseline
import openslide
import time
feature_extractor = resnet50_baseline(pretrained=True)
data_path = "/Volumes/HwangSSD2/TCGA_BLCA"
output_path = "/Users/M261759/Documents/GitHub/CLAM/example_bladder_output"#class1/features, class1/dictionary, class2/features, class2/dicitonaru
annotation_path = ""
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




def small_feature_extraction(slide_name, file_path, output_path, feature_extractor, tile_dims = [224,224], check_filter = True,copytarget = False):
    if copytarget:
        copytarget = './tempslide{}'.format(slide_name)
        copyfile(file_path, copytarget)
        slide = openslide.OpenSlide(copytarget)
    else:
        slide = openslide.OpenSlide(file_path)
    index_xy = {}
    slide_x, slide_y = slide.dimensions
    print("Working on: {}".format(slide_name))
    print("Whole Slide Dims: {}".format(slide.dimensions))
    tile_dim_x, tile_dim_y = tile_dims
    num_tiles_x, num_tiles_y =  (int)(slide_x/tile_dim_x), (int)(slide_y/tile_dim_y)
    total_tiles = num_tiles_y * num_tiles_x
    print("Number of tiles: {}".format(total_tiles))
    post_filter_tiles = 0
    for i_x in tqdm(range(num_tiles_x)):
        curr_x = i_x * tile_dim_x
        column_read = np.array(slide.read_region(location=(curr_x, 0), level = 1, size = (tile_dim_x, slide_y)))
        column_read = column_read[:,:,0:3]
        column_read = np.transpose(column_read,(1,0,2))
        for i_y in range(num_tiles_y):
            curr_y = i_y * tile_dim_y
            tile_curr = column_read[0:tile_dim_x, curr_y:curr_y+ tile_dim_y,:]
            if check_filter:
                if isBlackPatch_S(tile_curr, rgbThresh=20, percentage=0.05) or isWhitePatch_S(tile_curr, rgbThresh=220,
                                                                                          percentage=0.25):
                    continue

            index_xy.update({post_filter_tiles: (curr_x, curr_y)})
            post_filter_tiles += 1
    print("Post filter tiles: {}".format(post_filter_tiles))

    output = np.zeros(shape=(post_filter_tiles, 1024))
    for i in tqdm(range(post_filter_tiles)):
        curr_x, curr_y = index_xy[i]
        tile_curr = slide.read_region(location=(curr_x, curr_y), level=1, size=(tile_dim_x, tile_dim_y))
        tile_curr = np.array(tile_curr)[:, :, 0:3]
        tile_curr = torch.unsqueeze(torch.tensor(np.transpose(np.array(tile_curr),(2,0,1))),dim=0)/255
        #TODO batch loading
        if torch.cuda.is_available():
            tile_curr = tile_curr.cuda()
        features = feature_extractor(tile_curr)
        output[i, :] = features.cpu().detach().numpy()
    with open(os.path.join(output_path,'features',"{}.npy".format(slide_name)),'wb') as f:
        np.save(f, output)
    f = open(os.path.join(output_path,'dictionaries',"{}.dict".format(slide_name)), "wb")
    pickle.dump(index_xy, f)
    f.close()
    if copytarget:
        os.remove(copytarget)
    return output, index_xy




def simple_extraction(slide_name, file_path, output_path, feature_extractor, tile_dims = [224,224]):
    copytarget = './tempslide{}'.format(slide_name)
    copyfile(file_path, copytarget)
    slide = openslide.OpenSlide(copytarget)
    index_xy = {}
    slide_x, slide_y = slide.dimensions
    print("Working on: {}".format(slide_name))
    print("Whole Slide Dims: {}".format(slide.dimensions))
    tile_dim_x, tile_dim_y = tile_dims
    num_tiles_x, num_tiles_y =  (int)(slide_x/tile_dim_x), (int)(slide_y/tile_dim_y)
    total_tiles = num_tiles_y * num_tiles_x
    print("Number of tiles: {}".format(total_tiles))
    post_filter_tiles = 0
    output = np.zeros(shape=(num_tiles_x*num_tiles_y, 1024))
    for i_x in tqdm(range(num_tiles_x)):
        curr_x = i_x * tile_dim_x
        column_read = np.array(slide.read_region(location=(curr_x, 0), level = 1, size = (tile_dim_x, slide_y)))
        column_read = column_read[:,:,0:3]
        column_read = np.transpose(column_read,(1,0,2))
        print(column_read.shape)
        for i_y in range(num_tiles_y):
            curr_y = i_y * tile_dim_y
            tile_curr = column_read[0:tile_dim_x, curr_y:curr_y+ tile_dim_y,:]
            print(tile_curr.shape)
            tile_curr = torch.tensor(np.transpose(np.array(tile_curr), (2, 0, 1)))
            print(tile_curr.shape)
            tile_curr = torch.unsqueeze(tile_curr, dim=0) / 255
            # TODO batch loading
            #tile_curr = torch.tensor(np.array(tile_curr))/255
            print(tile_curr.shape)
            if torch.cuda.is_available():
                tile_curr = tile_curr.cuda()
            features = feature_extractor(tile_curr)
            output[post_filter_tiles, :] = features.cpu().detach().numpy()
            index_xy.update({post_filter_tiles: (curr_x, curr_y)})
            post_filter_tiles += 1

    print("Post filter tiles: {}".format(post_filter_tiles))
    with open(os.path.join(output_path,'features',"{}.npy".format(slide_name)),'wb') as f:
        np.save(f, output)
    f = open(os.path.join(output_path,'dictionaries',"{}.dict".format(slide_name)), "wb")
    pickle.dump(index_xy, f)
    f.close()
    os.remove(copytarget)
    return output, index_xy




def small_feature_extraction_high_mem(slide_name, file_path, output_path, feature_extractor, tile_dims = [224,224], check_filter = True,copytarget = False):
    if copytarget:
        copytarget = './tempslide{}'.format(slide_name)
        copyfile(file_path, copytarget)
        slide = openslide.OpenSlide(copytarget)
    else:
        slide = openslide.OpenSlide(file_path)
    index_xy = {}
    slide_x, slide_y = slide.dimensions
    print("Working on: {}".format(slide_name))
    print("Whole Slide Dims: {}".format(slide.dimensions))
    tile_dim_x, tile_dim_y = tile_dims
    num_tiles_x, num_tiles_y =  (int)(slide_x/tile_dim_x), (int)(slide_y/tile_dim_y)
    total_tiles = num_tiles_y * num_tiles_x
    print("Number of tiles: {}".format(total_tiles))
    post_filter_tiles = 0
    whole_slide_read = np.array(slide.read_region(location=(0,0)), level = 1, size = (slide_x, slide_y))
    whole_slide_read = whole_slide_read[:,:,0:3]
    for i_x in tqdm(range(num_tiles_x)):
        for i_y in range(num_tiles_y):
            curr_y = i_y * tile_dim_y
            curr_x = i_x * tile_dim_x
            tile_curr = whole_slide_read[curr_x:curr_x+tile_dim_x, curr_y:curr_y + tile_dim_y,:]
            if check_filter:
                if isBlackPatch_S(tile_curr, rgbThresh=20, percentage=0.05) or isWhitePatch_S(tile_curr, rgbThresh=220,
                                                                                          percentage=0.25):
                    continue

            index_xy.update({post_filter_tiles: (curr_x, curr_y)})
            post_filter_tiles += 1
    print("Post filter tiles: {}".format(post_filter_tiles))

    output = np.zeros(shape=(post_filter_tiles, 1024))
    for i in tqdm(range(post_filter_tiles)):
        curr_x, curr_y = index_xy[i]
        tile_curr = whole_slide_read[curr_x:curr_x+tile_dim_x, curr_y:curr_y + tile_dim_y,:]
        tile_curr = np.array(tile_curr)[:, :, 0:3]
        tile_curr = torch.unsqueeze(torch.tensor(np.transpose(np.array(tile_curr),(2,0,1))),dim=0)/255
        #TODO batch loading
        if torch.cuda.is_available():
            tile_curr = tile_curr.cuda()
        features = feature_extractor(tile_curr)
        output[i, :] = features.cpu().detach().numpy()
    with open(os.path.join(output_path,'features',"{}.npy".format(slide_name)),'wb') as f:
        np.save(f, output)
    f = open(os.path.join(output_path,'dictionaries',"{}.dict".format(slide_name)), "wb")
    pickle.dump(index_xy, f)
    f.close()
    if copytarget:
        os.remove(copytarget)
    return output, index_xy