import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import ndimage
from tqdm import tqdm
import scipy.sparse as sparse
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from matplotlib import rcParams

def set_all_rcParams(true_or_false):
    rcParams['ytick.left']=true_or_false
    rcParams['xtick.bottom']=true_or_false
    rcParams['ytick.labelleft'] = true_or_false
    rcParams['xtick.labelbottom'] = true_or_false

def plot_for_gif(image_to_save,num_iter, i):
    fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':[8, 1]}, figsize=(14,10))
    ax[0].imshow(image_to_save, cmap='viridis')
    ax[0].axis('off')
    ax[1].axvline(x=.5, c='k')
    ax[1].scatter(.5, i, c='k')
    ax[1].set_ylim([num_iter, 0])
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    # ax[1].xticks([], [])
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    plt.subplots_adjust(wspace=.04, hspace=0)
    plt.savefig(f'{path_img_dest}images before gifs/iter {i:5d}.jpeg',
                bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)
    
def save_original(image_to_save, id_name, name_extension, error_final=-1):
    name_extension = str(name_extension)
    fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':[8, 1]}, figsize=(14,10))
    ax[0].imshow(image_to_save, cmap='viridis')
    ax[0].axis('off')
    ax[1].axvline(x=.5, c='k')
    ax[1].set_ylim([num_iter, 0])
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    plt.subplots_adjust(wspace=.04, hspace=0)
    if error_final==-1: # for original
        fig.savefig(f'{path_img_dest}gifs/dip {id_name} {name_extension}.jpeg',
                    bbox_inches = 'tight',pad_inches = 0)
    else:
        fig.savefig(f'{path_img_dest}gifs/dip {id_name} {name_extension} {error_final:05d}.jpeg',
                    bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)

def plot_3d(image, threshold=-300, alpha=.70, fig_size=10):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, x,y = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def plot_3d_2(image, image2, threshold=-300, threshold2=-300, alpha=.70, fig_size=10):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    verts, faces, x,y = measure.marching_cubes_lewiner(p, threshold)
    
    p2 = image2.transpose(2,1,0)
    verts2, faces2, x2,y2 = measure.marching_cubes_lewiner(p2, threshold2)

    fig = plt.figure(figsize=(fig_size*2, fig_size))
    ax = fig.add_subplot(121, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    ax = fig.add_subplot(122, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts2[faces2], alpha=alpha)
    face_color = [0.75, 0.25, 0.25]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def read_slices(new_name):
    """Read slices of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    idname = new_name.split('_')[0]
    file_lung = np.load(f'{path_data}lungs/{new_name}')
    file_mask = np.load(f'{path_data}masks/{new_name}')
    file_nodule = np.load(f'{path_data}nodule to focus on/{new_name}')
    file_outside  = np.load(f'{path_data}outside lungs mask/{new_name}')
    lungs_slice = file_lung.f.arr_0
    mask_slice = file_mask.f.arr_0
    nodule = file_nodule.f.arr_0
    outside_lungs = file_outside.f.arr_0
    return lungs_slice, mask_slice, nodule, outside_lungs

def make3d_from_sparse(path):
    slices_all = os.listdir(path)
    slices_all = np.sort(slices_all)
    for idx, i in tqdm(enumerate(slices_all), desc='reading slices', total=len(slices_all)):
        sparse_matrix = sparse.load_npz(f'{path}{i}')
        array2d = np.asarray(sparse_matrix.todense())
        if idx == 0: 
            scan3d = array2d
            continue
        scan3d = np.dstack([scan3d,array2d])
    return scan3d

def make_images_right_size3D(lungs_slice, mask_slice, mask_maxvol_and_lungs_small, outside_lungs):
    """Make the images the right size 
    The encoder-decoder has five blocks (the one initially evalluated), 
    therefore, each side has to be divisible by a factor of 32 (2^5)"""
    print('formating shape')
    factor = 32
    pad_dim_0 = factor - np.shape(lungs_slice)[0] % factor
    pad_dim_1 = factor - np.shape(lungs_slice)[1] % factor
    pad_dim_2 = factor - np.shape(lungs_slice)[2] % factor

    #mask_slice = 1 - mask_slice

    lungs_slice = np.pad(lungs_slice, ((0,pad_dim_0), (0,pad_dim_1), (0, pad_dim_2)), mode='constant')
    mask_slice = np.pad(mask_slice, ((0,pad_dim_0), (0,pad_dim_1), (0, pad_dim_2)), mode='constant')
    mask_max =  np.pad(mask_maxvol_and_lungs_small, ((0,pad_dim_0), (0,pad_dim_1), (0, pad_dim_2)), mode='constant')
    outside_lungs = np.pad(outside_lungs, ((0,pad_dim_0), (0,pad_dim_1), (0, pad_dim_2)), mode='constant', constant_values=0)

    # Normalize
    lungs_slice = (lungs_slice - np.min(lungs_slice))/(np.max(lungs_slice)-np.min(lungs_slice))
    
    # Add dimensions
    lungs_slice = np.expand_dims(lungs_slice, 0)
    mask_slice = np.expand_dims(mask_slice, 0)
    outside_lungs = np.expand_dims(outside_lungs, 0)
    mask_max = np.expand_dims(mask_max, 0)
    

    img_np = lungs_slice
    img_mask_np = mask_max
    return img_np, img_mask_np, outside_lungs

def read_slices3D(path_data, ii_ids):
    """Read VOLUMES of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    #ii_ids = f'LIDC-IDRI-{idnumber:04d}'
    print(f'reading scan {ii_ids}')
    vol = make3d_from_sparse(f'{path_data}{ii_ids}/scans/')
    mask = make3d_from_sparse(f'{path_data}{ii_ids}/consensus_masks/')
    mask_maxvol = make3d_from_sparse(f'{path_data}{ii_ids}/maxvol_masks/')
    mask_lungs = make3d_from_sparse(f'{path_data}{ii_ids}/lung_masks/')  
    # rearrange axes to slices first
    vol = np.swapaxes(vol,1,2)
    vol = np.swapaxes(vol,0,1)
    mask = np.swapaxes(mask,1,2)
    mask = np.swapaxes(mask,0,1)
    mask_maxvol = np.swapaxes(mask_maxvol,1,2)
    mask_maxvol = np.swapaxes(mask_maxvol,0,1)
    mask_lungs = np.swapaxes(mask_lungs,1,2)
    mask_lungs = np.swapaxes(mask_lungs,0,1)
    # Find the minimum box that contain the lungs 
    min_box = np.where(vol!=0)
    min_box_c = min_box[0]
    min_box_x = min_box[1]
    min_box_y = min_box[2]
    # Apply the minimum box to the vol and masks
    vol_small = vol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_small = mask[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_maxvol_small = mask_maxvol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_lungs_small = mask_lungs[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)] 
    # Get the mask_maxvol_small and the mask_lungs_small together
    mask_maxvol_and_lungs = mask_lungs_small - mask_maxvol_small
    return vol_small, mask_maxvol_small, mask_maxvol_and_lungs, mask_lungs_small

def read_slices3D_v2(path_data, ii_ids):
    """Read VOLUMES of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    #ii_ids = f'LIDC-IDRI-{idnumber:04d}'
    print(f'reading scan {ii_ids}')
    vol = make3d_from_sparse(f'{path_data}{ii_ids}/scans/')
    mask = make3d_from_sparse(f'{path_data}{ii_ids}/consensus_masks/')
    mask_maxvol = make3d_from_sparse(f'{path_data}{ii_ids}/maxvol_masks/')
    mask_lungs = make3d_from_sparse(f'{path_data}{ii_ids}/lung_masks/')  
    # rearrange axes to slices first
    vol = np.swapaxes(vol,1,2)
    vol = np.swapaxes(vol,0,1)
    mask = np.swapaxes(mask,1,2)
    mask = np.swapaxes(mask,0,1)
    mask_maxvol = np.swapaxes(mask_maxvol,1,2)
    mask_maxvol = np.swapaxes(mask_maxvol,0,1)
    mask_lungs = np.swapaxes(mask_lungs,1,2)
    mask_lungs = np.swapaxes(mask_lungs,0,1)
    # Find the minimum box that contain the lungs 
    min_box = np.where(vol!=0)
    min_box_c = min_box[0]
    min_box_x = min_box[1]
    min_box_y = min_box[2]
    # Apply the minimum box to the vol and masks
    vol_small = vol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_small = mask[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_maxvol_small = mask_maxvol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_lungs_small = mask_lungs[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)] 
    # Get the mask_maxvol_small and the mask_lungs_small together
    mask_maxvol_and_lungs = 1- ((1-mask_lungs_small) | mask_maxvol_small)
    mask_lungs_small2 = mask_lungs_small | mask_maxvol_small
    return vol_small, mask_maxvol_small, mask_maxvol_and_lungs, mask_lungs_small2

def erode_and_split_mask(mask_lungs, slice_middle):
    '''We return the center of each lung (from the middle slice). We erode the center slice of the
    lungs mask to have the lungs separated.'''
    # Erode mask
    mask_center_slice = mask_lungs[slice_middle,:,:]
    mask_slice_eroded = binary_erosion(mask_center_slice, iterations=10)
    # Rectangle for lung 1
    labeled, nr_objects = ndimage.label(mask_slice_eroded) 
    blank = np.zeros_like(labeled)
    x, y = np.where(labeled==2)
    blank[x,y] = 2
    ymed_1 = np.median(y); xmed_1 = np.median(x)
    #coords_i_1, coords_j_1, coords_k_1 = find_best_vol(mask_lungs, xmed_1, ymed_1, side1, side2, side3)
    # Rectangle for lung 2
    labeled, nr_objects = ndimage.label(mask_slice_eroded) 
    blank = np.zeros_like(labeled)
    x, y = np.where(labeled==1)
    blank[x,y] = 1
    ymed_2 = np.median(y); xmed_2 = np.median(x)
    # Make sure that number 1 is the lung in the left
    if ymed_1 > ymed_2:
        ymed_temp = ymed_1
        xmed_temp = xmed_1
        ymed_1 = ymed_2
        xmed_1 = xmed_2
        ymed_2 = ymed_temp
        xmed_2 = xmed_temp
    return xmed_1, ymed_1, xmed_2, ymed_2

def box_coords_contain_masks_right_size(coord_max_sideX, coord_min_sideX):
    
    # Max and min coord of nodules for each axis
    z_max_sideX = np.max(np.array(coord_max_sideX)[:,0])
    z_min_sideX = np.min(np.array(coord_min_sideX)[:,0])
    x_max_sideX = np.max(np.array(coord_max_sideX)[:,1])
    x_min_sideX = np.min(np.array(coord_min_sideX)[:,1])
    y_max_sideX = np.max(np.array(coord_max_sideX)[:,2])
    y_min_sideX = np.min(np.array(coord_min_sideX)[:,2])

    # find out the length required to contain all masks per axis
    z_dist_required = z_max_sideX - z_min_sideX
    x_dist_required = x_max_sideX - x_min_sideX
    y_dist_required = y_max_sideX - y_min_sideX
    
    # Fixed distance
    z_dist_adjusted = 96
    x_dist_adjusted = 160
    y_dist_adjusted = 96

    # Add half of the required length to min, and then, get the new max using the required length 
    #add_one_side_z = (factor - z_dist_required % factor)//2
    add_one_side_z = (z_dist_adjusted - z_dist_required)//2
    z_min_sideX  = int(z_min_sideX - add_one_side_z)
    z_min_sideX = np.max([z_min_sideX, 0]) # check it's not smaller than 0
    z_max_sideX_temp = z_min_sideX + z_dist_adjusted
    if z_max_sideX_temp > np.shape(mask_lungs_small)[0]: # if max is outside the scan
        z_min_sideX = z_max_sideX - z_dist_adjusted
    else:
        z_max_sideX = z_max_sideX_temp
    
    #add_one_side_x = (factor - x_dist_required % factor)//2
    add_one_side_x = (x_dist_adjusted - x_dist_required)//2
    x_min_sideX  = int(x_min_sideX - add_one_side_x)
    x_min_sideX = np.max([x_min_sideX, 0])
    x_max_sideX_temp = x_min_sideX + x_dist_adjusted
    if x_max_sideX_temp > np.shape(mask_lungs_small)[1]: # if max is outside the scan
        x_min_sideX = x_max_sideX - x_dist_adjusted
    else:
        x_max_sideX = x_max_sideX_temp

    #add_one_side_y = (factor - y_dist_required % factor)//2
    add_one_side_y = (y_dist_adjusted - y_dist_required)//2
    y_min_sideX  = int(y_min_sideX - add_one_side_y)
    y_min_sideX = np.max([y_min_sideX, 0])
    y_max_sideX_temp = y_min_sideX + y_dist_adjusted
    if y_max_sideX_temp > np.shape(mask_lungs_small)[2]: # if max is outside the scan
        y_min_sideX = y_max_sideX - y_dist_adjusted
    else:
        y_max_sideX = y_max_sideX_temp

    return z_min_sideX, z_max_sideX, x_min_sideX, x_max_sideX, y_min_sideX, y_max_sideX

def box_coords_contain_masks_right_size_search(coord_max_sideX, coord_min_sideX, side, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, dist1 = 96, dist2 = 160, dist3 = 96):
    # new shapes are defined with distances on each axes
    length1 = dist1//2
    length2 = dist2//2
    length3 = dist3//2
    # limits of the nodules masks
    if len(coord_max_sideX) > 0:
        coord_ = [i[0] for i in coord_max_sideX]
        z_max_sideX = np.max(coord_)
        coord_ = [i[0] for i in coord_min_sideX]
        z_min_sideX = np.min(coord_)
        coord_ = [i[1] for i in coord_max_sideX]
        x_max_sideX = np.max(coord_)
        coord_ = [i[1] for i in coord_min_sideX]
        x_min_sideX = np.min(coord_)
        coord_ = [i[2] for i in coord_max_sideX]
        y_max_sideX = np.max(coord_)
        coord_ = [i[2] for i in coord_min_sideX]
        y_min_sideX = np.min(coord_)

    # find if the coords are closer to the center of the right or left lung
    if side == 1:
        xmed_X = xmed_1
        ymed_X = ymed_1
    elif side == 2:
        xmed_X = xmed_2
        ymed_X = ymed_2
    box_found = False  
    
    # find where the vol_cut get more info voxels
    max_sum = 0
    for i in range(30):
        ii = i * 4 - 58
        for j in range(19):
            jj = j * 3 - 27
            for k in range(19):
                kk = k * 4 - 36
            
                
                #if ii == 0 and jj == 0 and kk == 0: pdb.set_trace()
                #zmin = int(slice_middle-length1+ii); zmax = int(slice_middle+length1+ii)
                #xmin = int(xmed_X-length2+jj); xmax = int(xmed_X+length2+jj)
                #ymin = int(ymed_X-length3+kk); ymax = int(ymed_X+length3+kk)
                
                # limits of the current box
                zmin = int(slice_middle-(dist1//2)+ii)
                zmin = np.max([zmin, 0]); zmax = int(zmin + dist1)
                
                xmin = int(xmed_X-(dist2//2)+jj); 
                xmin = np.max([xmin, 0]); xmax = int(xmin + dist2)
                
                ymin = int(ymed_X-(dist3//2)+kk); 
                ymin = np.max([ymin, 0]); ymax = int(ymin + dist3)
            
                #max_cut = mask_maxvol_small[zmin:zmax, xmin:xmax, zmin:zmax]
            
                #if there is a nodule
                if len(coord_max_sideX) > 0:
                    #if the current box contains the masks
                    if zmin < z_min_sideX and zmax > z_max_sideX and xmin < x_min_sideX and xmax > x_max_sideX and ymin < y_min_sideX and ymax > y_max_sideX:
                        #if the current box is inside the scan (small) limits
                        if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                            vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                            # the box contains as many info voxels as possible
                            this_sum = np.sum(vol_cut)
                            if this_sum > max_sum:
                                max_sum = this_sum
                                coords_i = ii; coords_j=jj; coords_k=kk
                                box_found = True
                                z_min_sideX_found = zmin
                                z_max_sideX_found = zmax
                                x_min_sideX_found = xmin
                                x_max_sideX_found = xmax
                                y_min_sideX_found = ymin                        
                                y_max_sideX_found = ymax 
                else: # if it doesn't contain the masks just look for max value of info voxels
                    vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                    #if the current box is inside the scan (small) limits
                    if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                        # the box contains as many info voxels as possible
                        this_sum = np.sum(vol_cut)
                        if this_sum >= max_sum:
                            max_sum = this_sum
                            coords_i = ii; coords_j=jj; coords_k=kk
                            box_found = True
                            z_min_sideX_found = zmin
                            z_max_sideX_found = zmax
                            x_min_sideX_found = xmin
                            x_max_sideX_found = xmax
                            y_min_sideX_found = ymin                        
                            y_max_sideX_found = ymax 
            #print(int(zmin < z_min_sideX) , int(zmax > z_max_sideX) , int(xmin < x_min_sideX) , int(xmax > x_max_sideX) , int(ymin < y_min_sideX) , int(ymax > y_max_sideX))
    if box_found == True:
        return z_min_sideX_found, z_max_sideX_found, x_min_sideX_found, x_max_sideX_found, y_min_sideX_found, y_max_sideX_found
    else:
        return -1, -1, -1, -1, -1, -1

def nodule_right_or_left_lung(mask_maxvol_smallX, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2):
    '''For each nodule determine if its closer to the right or left cube center.
    Then return, for each side, the min and max coordianates of each nodule'''
    labeled, nr_objects = ndimage.label(mask_maxvol_smallX) 
    masks_ndl = [np.where(labeled==i) for i in range(nr_objects+1) if i>0]   # masks for individual nodules masks
    masks_ndl_centers = [np.median(i,1) for i in masks_ndl] # centers individual nodules masks
    masks_ndl_max = [np.max(i,1) for i in masks_ndl] # centers individual nodules masks
    masks_ndl_min = [np.min(i,1) for i in masks_ndl] # centers individual nodules masks
    
    # For each nodule determine if its closer to the right or left cube center
    nodule_in_side = np.ones((len(masks_ndl_centers)))
    center1 = (slice_middle,xmed_1,ymed_1)
    center2 = (slice_middle,xmed_2,ymed_2)
    for idx, i in enumerate(masks_ndl_centers):
        dist1 = np.linalg.norm(center1-i)
        dist2 = np.linalg.norm(center2-i)
        if dist2 < dist1:
            nodule_in_side[idx]=2
            
    coord_center_side1_X, coord_max_side1_X, coord_min_side1_X = [], [], []
    coord_center_side2_X, coord_max_side2_X, coord_min_side2_X = [], [], []
    for coords, coords_max, coords_min, side in zip(masks_ndl_centers, masks_ndl_max, masks_ndl_min, nodule_in_side):
        if side ==1:
            coord_center_side1_X.append(coords)
            coord_max_side1_X.append(coords_max)
            coord_min_side1_X.append(coords_min)
        if side == 2:
            coord_center_side2_X.append(coords)
            coord_max_side2_X.append(coords_max)
            coord_min_side2_X.append(coords_min)
    return coord_min_side1_X, coord_max_side1_X, coord_min_side2_X, coord_max_side2_X

# https://stackoverflow.com/questions/49515085/python-garbage-collection-sometimes-not-working-in-jupyter-notebook
def my_reset(*varnames):
    """
    varnames are what you want to keep
    """
    globals_ = globals()
    to_save = {v: globals_[v] for v in varnames}
    to_save['my_reset'] = my_reset  # lets keep this function by default
    del globals_
    get_ipython().magic("reset -f")
    globals().update(to_save)

def get_block_if_ndl(block1, block2, block1_mask, block2_mask, block1_mask_maxvol_and_lungs, block2_mask_maxvol_and_lungs, block1_mask_lungs, block2_mask_lungs):
    '''If there are nodules in both blocks put them in a list to be processed on be one in a loop.
    Also include their mask and their names for identification'''
    blocks_ndl, blocks_ndl_mask, blocks_ndl_lungs_mask, block_mask_maxvol_and_lungs = [], [], [], []
    blocks_ndl_names = []
    z,x,y = np.where(block1_mask==1)
    if len(z)>1:
        slice1 = int(np.median(z))
        blocks_ndl.append(block1)
        blocks_ndl_mask.append(block1_mask)
        blocks_ndl_lungs_mask.append(block1_mask_lungs)
        block_mask_maxvol_and_lungs.append(block1_mask_maxvol_and_lungs)
        blocks_ndl_names.append('block1')
    else:
        slice1 = np.shape(block1_mask)[0]//2

    z,x,y = np.where(block2_mask==1)
    if len(z)>1:
        slice2 = int(np.median(z))
        blocks_ndl.append(block2)
        blocks_ndl_mask.append(block2_mask)
        blocks_ndl_lungs_mask.append(block2_mask_lungs)
        block_mask_maxvol_and_lungs.append(block2_mask_maxvol_and_lungs)
        blocks_ndl_names.append('block2')
    else:
        slice2 = np.shape(block2_mask)[0]//2
    return blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs_mask, blocks_ndl_names, slice1, slice2

def get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zmin, c_zmax, c_xmin, c_xmax, c_ymin, c_ymax):
    '''Based on the limits found in "box_coords_contain_masks_right_size_search"
    get the block for the lung scan, the block for the mask with the maxvol segmentation,
    the block for the mask with the maxvol segmentation and the lungs and the block
    for the mask with the mask of the lungs'''
    block = vol_small[c_zmin:c_zmax, c_xmin:c_xmax, c_ymin:c_ymax]
    block_mask = mask_maxvol_small[c_zmin:c_zmax, c_xmin:c_xmax, c_ymin:c_ymax]
    block_mask_maxvol_and_lungs = mask_maxvol_and_lungs_small[c_zmin:c_zmax, c_xmin:c_xmax, c_ymin:c_ymax]
    block_mask_lungs = mask_lungs_small[c_zmin:c_zmax, c_xmin:c_xmax, c_ymin:c_ymax]
    return block, block_mask, block_mask_maxvol_and_lungs, block_mask_lungs