import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.optim
import imageio
from copy import copy
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from tqdm import tqdm_notebook
from skimage import measure, morphology
from itertools import groupby, count
import matplotlib.patches as patches
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from torch.autograd import Variable

from models.resnet import ResNet
from models.unet import UNet
from models.skip3D import skip
from utils.inpainting_utils3D import *

from inpainting_nodules_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('skip_idx', type=int, help='skip indices already processed')
args = parser.parse_args()
print(args.skip_idx)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
PLOT = True
imsize = -1
dim_div_by = 64

path_data = f'/data/OMM/Datasets/LIDC_other_formats/LIDC_preprocessed_3D v4 - inpaint before preprocess/'
path_data_old = '/data/OMM/Datasets/LIDC_other_formats/LIDC_preprocessed_3D v2/'
path_img_dest = '/data/OMM/project results/Feb 5 19 - Deep image prior/dip results all 15/'

torch.cuda.set_device(0)
torch.cuda.empty_cache()

dtype = torch.cuda.FloatTensor
from torch.autograd import Variable

# NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet
pad = 'zero' # 'zero' OMM it was reflection
OPT_OVER = 'net'
OPTIMIZER = 'adam'
INPUT = 'noise'
input_depth = 32
#LR = 0.000001 
num_iter = 10001 # 10001
param_noise = False
show_every = 500
figsize = 5
reg_noise_std = 0.00

# This section has to be the same as the one defined in common_utils3D
LR = 1e-7
LRs = []
for i in range(1000):
    LR *= 1.2 #1.1
    if LR >= 1e-2: break
    LRs.append(LR)

def closure():
    global i
    images_all = []
    
    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50
    
    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
    out = net(net_input)
#     print(np.shape(out))
    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()
        
    print ('Iteration %05d    Loss %.12f' % (i, total_loss.item()), '\r', end='')
    #if  PLOT and i % show_every == 0:
    if  PLOT:
        out_np = torch_to_np(out)
        if np.shape(out_np)[0] == 1:
            image_to_save = out_np[0]
        #plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1) # DEL original fun
        #plot_for_gif(image_to_save, num_iter, i) # DEL save images to make gif
        images_all.append(image_to_save)
        
    i += 1    
#     if  PLOT and i % show_every == 0: image_generated = image_to_save
#     else: image_generated = []
    
    return total_loss, images_all

ids = os.listdir(path_data)
ids = np.sort(ids)
for idx_name, name in enumerate(ids):
    torch.cuda.empty_cache()
    if idx_name < args.skip_idx:continue
    print('')
    print(f'{name}, ({idx_name})')
    vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small = read_slices3D_v2(path_data, name)
    slice_middle = np.shape(vol_small)[0] // 2
    xmed_1, ymed_1, xmed_2, ymed_2 = erode_and_split_mask(mask_lungs_small,slice_middle)
    coord_min_side1, coord_max_side1, coord_min_side2, coord_max_side2 = nodule_right_or_left_lung(mask_maxvol_small, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2)
    c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2 = box_coords_contain_masks_right_size_search(coord_max_side2, coord_min_side2, 2, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small)
    c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1 = box_coords_contain_masks_right_size_search(coord_max_side1, coord_min_side1, 1,  slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small)
    # Block1 and Block2: lungs, ndl mask, lungs mask, ndl&lungs mask
    block1, block1_mask, block1_mask_maxvol_and_lungs, block1_mask_lungs = get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zmin1, c_zmax1, c_xmin1, c_xmax1, c_ymin1, c_ymax1)
    block2, block2_mask, block2_mask_maxvol_and_lungs, block2_mask_lungs = get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zmin2, c_zmax2, c_xmin2, c_xmax2, c_ymin2, c_ymax2)
    # Normalization is applied using the min and max of all images
    block1 = (block1 - (-1018.0))/(1171.0-(-1018.0)) 
    block1 = np.clip(block1,0,1)
    block2 = (block2 - (-1018.0))/(1171.0-(-1018.0)) 
    block2 = np.clip(block2,0,1)
    # Apply lungs' mask
    block1 = block1*block1_mask_lungs
    block2 = block2*block2_mask_lungs
    # Get those blocks where there is a nodule in
    blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names, slice1, slice2 =  get_block_if_ndl(block1, block2, block1_mask, block2_mask, block1_mask_maxvol_and_lungs, block2_mask_maxvol_and_lungs, block1_mask_lungs, block2_mask_lungs)
    # delete variables
    del block1, block2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small
    for (block, block_mask, block_maxvol_and_lungs, block_lungs, block_name) in zip(blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names): 
        torch.cuda.empty_cache()
        print(block_name)
        # Add batch channels
        img_np = np.expand_dims(block,0)
        img_mask_np = np.expand_dims(block_maxvol_and_lungs,0)

        # LR FINDER
        restart_i = 0
        restart = True

        mse = torch.nn.MSELoss().type(dtype)

        # with torch.no_grad():

        img_var = Variable(np_to_torch(img_np).type(dtype))
        mask_var = Variable(np_to_torch(img_mask_np).type(dtype))
        # img_var = torch.tensor(img_np, requires_grad=False).type(dtype)
        # mask_var = torch.tensor(img_mask_np, requires_grad=False).type(dtype)

        # LR finder

        net = skip(input_depth, img_np.shape[0], 
                num_channels_down = [128] * 5,
                num_channels_up   = [128] * 5,
                num_channels_skip = [0] * 5,
                upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

        net = net.type(dtype)

        net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)


        mse_error = []
        start = time.time()
        i = 0

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        p = get_params(OPT_OVER, net, net_input)

        mse_error, images_generated_all, best_iter = optimize4(OPTIMIZER, p, closure, LR, num_iter, show_every, path_img_dest, restart, annealing=False, lr_finder_flag=True)
        mse_error_lr = mse_error
        # mse_error_lr = [i.detach().cpu().numpy() for i in mse_error]
        mse_error_lr = np.squeeze(mse_error_lr)
        #mse_error_lr_all.append(mse_error_lr)

        del net

        # Find the longest sequence of slope < -1e-4
        loss_going_down = np.where(np.diff(mse_error_lr) < -1e-7) # indices that go down (negative diff)
        loss_going_down = list(loss_going_down[0] + 1) # for each pair of indices with neg diff take the 2nd one and convert to list
        c = count()
        val = max((list(g) for _, g in groupby(loss_going_down, lambda x: x-next(c))), key=len) # longest sequence of negative diff
        val = list(val)
        slope_diff = np.diff(mse_error_lr[val])
        largest_diff = np.where(slope_diff == np.min(slope_diff))[0]
        LR = LRs[val[largest_diff[0]]]
        np.save(f'{path_img_dest}learning rates/{name}_{block_name}.npy',LR)
        np.save(f'{path_img_dest}learning rates/mse error curves/{name}_{block_name}.npy',mse_error_lr)