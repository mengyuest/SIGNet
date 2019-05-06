# Mostly based on the code written by Clement Godard: 
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
from __future__ import division
import sys
import cv2
import os
import numpy as np
import argparse
from depth_evaluation_utils import *
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default='eigen', help='eigen or stereo split')
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
parser.add_argument('--vis_dir', type=str, help='Path to the visualization output directory')
parser.add_argument('--interp', default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to interpolation depth")
parser.add_argument('--vis_limit',type=int, default=-1, help="How many images processing. -1 for no limits")
parser.add_argument('--show_vis',default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to show visualization")
parser.add_argument('--mask_eval',default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to also show masked evaluation")
parser.add_argument('--mask_kitti_dir',default="",type=str,help="Where for the kitti test dataset semantic results")
parser.add_argument('--mask_suffix',default="", type=str, help="The suffix for semantic files. Could be just '.npy' ")
parser.add_argument('--mask_by_channel_start_index',default="",type=str,help="-1 for all, 0<=x<=18 for a specific label")
parser.add_argument('--mask_by_channel_end_index',default="",type=str,help="-1 for all, 0<=x<=18 for a specific label")

args = parser.parse_args()

with open('data/kitti/test_files_%s.txt' % args.split, 'r') as f:
    ori_test_files = f.readlines()
    ori_test_files = [args.kitti_dir + t[:-1] for t in ori_test_files]

def convert_disps_to_depths_stereo(gt_disparities, pred_depths):
    gt_depths = []
    pred_depths_resized = []
    pred_disparities_resized = []
    
    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(1./pred_depth) 

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))

        gt_depths.append(gt_depth)
        pred_depths_resized.append(pred_depth)
    return gt_depths, pred_depths_resized, pred_disparities_resized

def pltsave(img,img_dir,depth_vis=True):
    if depth_vis:
        plt.imsave(img_dir, 1/img, cmap="plasma")
    else:
        plt.imsave(img_dir, img)

def main():
    t1=time.time()
    load_gt_from_file=False
    if args.interp:
        load_gt_path="./models/gt_data/gt_depth_interp.npy"
    else:
        load_gt_path="./models/gt_data/gt_depth.npy"
        load_gt_interp_path="./models/gt_data/gt_depth_interp.npy"

    if os.path.exists(load_gt_path):
        load_gt_from_file=True
        loaded_gt_depths=np.load(load_gt_path)
        loaded_gt_interps=np.load(load_gt_interp_path)
    pred_depths = np.load(args.pred_file)

    args.test_file_list = './data/kitti/test_files_%s.txt' % args.split

    test_files=[]

    if args.split == 'eigen':
        test_files = read_text_lines(args.test_file_list)
        if load_gt_from_file:
            num_test=len(loaded_gt_depths)
        else:
            gt_files, gt_calib, im_sizes, im_files, cams = \
                read_file_data(test_files, args.kitti_dir)
            num_test=len(im_files)
        if args.vis_limit>0:
           num_test=args.vis_limit
        gt_depths = []
        gt_interps=[]
        pred_depths_resized = []
        for t_id in range(num_test):

            if load_gt_from_file:

                img_size_h,img_size_w=loaded_gt_depths[t_id].shape
                depth = loaded_gt_depths[t_id]
                interp = loaded_gt_interps[t_id]

            else:
                img_size_h=im_sizes[t_id][0]
                img_size_w=im_sizes[t_id][1]
                camera_id = cams[t_id]  # 2 is left, 3 is right
                depth,interp,velo_pts = generate_depth_map(gt_calib[t_id], 
                                       gt_files[t_id], 
                                       im_sizes[t_id], 
                                       camera_id, 
                                       args.interp, 
                                       True)
                
                if args.interp:
                    depth = interp # in this case, gen_...() returns depth, depth_interp
            gt_depths.append(depth.astype(np.float32))
            gt_interps.append(interp.astype(np.float32))

            pred_depths_resized.append(
                cv2.resize(pred_depths[t_id], 
                        (img_size_w, img_size_h), 
                        interpolation=cv2.INTER_LINEAR))


        if load_gt_from_file==False:
            os.makedirs("./models/gt_data", exist_ok=True)
            np.save(load_gt_path, gt_depths)
            np.save(load_gt_interp_path, gt_interps)
        
        pred_depths = pred_depths_resized
    else:
        num_test = 200
        gt_disparities = load_gt_disp_kitti(args.kitti_dir)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_stereo(gt_disparities, pred_depths)

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    
    #TODO read semantic segs
    if args.mask_eval:
        bg_rms     = np.zeros(num_test, np.float32)
        bg_log_rms = np.zeros(num_test, np.float32)
        bg_abs_rel = np.zeros(num_test, np.float32)
        bg_sq_rel  = np.zeros(num_test, np.float32)
        bg_d1_all  = np.zeros(num_test, np.float32)
        bg_a1      = np.zeros(num_test, np.float32)
        bg_a2      = np.zeros(num_test, np.float32)
        bg_a3      = np.zeros(num_test, np.float32)
        bg_masks=[]

        for i,line in enumerate(test_files):
            # naming style and loading method
            sem_seg = np.load(args.mask_kitti_dir+line[:-4]+args.mask_suffix) 
            #TODO fetch the background
            if args.mask_by_channel_start_index=="":
                bg_masks.append((sem_seg<=10).astype(np.uint8))
            else:
                mask_values = list(map(int,args.mask_by_channel_start_index.strip().split(",")))
                bg_masks.append((np.isin(sem_seg,mask_values)).astype(np.uint8))
                
    for i in range(num_test):    
        gt_depth = gt_depths[i]
        gt_interp = gt_interps[i]
        pred_depth = np.copy(pred_depths[i])
        
        if args.split == 'eigen':
            
            mask = np.logical_and(gt_depth > args.min_depth, 
                                  gt_depth < args.max_depth)
            # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            gt_height, gt_width = gt_depth.shape

            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                            0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            

        if args.split == 'stereo':
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        #TODO semantic mask (fetch only background)
        if args.mask_eval:
            bg_pred_depth = np.copy(pred_depths[i])
            bg_mask = bg_masks[i]
            bg_mask = np.logical_and(mask, bg_mask)

        # Scale matching
        scalar = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        pred_depth[mask] *= scalar

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
        compute_errors(gt_depth[mask], pred_depth[mask])
        
        #TODO semantic masked evaulation
        if args.mask_eval:
            bg_pred_depth[mask] *= scalar
            bg_pred_depth[bg_pred_depth < args.min_depth] = args.min_depth
            bg_pred_depth[bg_pred_depth > args.max_depth] = args.max_depth
            bg_abs_rel[i], bg_sq_rel[i], bg_rms[i], bg_log_rms[i], bg_a1[i], bg_a2[i], bg_a3[i] = \
                compute_errors(gt_depth[bg_mask], bg_pred_depth[bg_mask])
            
    print('Evaluating {} took {:.4f} secs'.format(args.pred_file, time.time()-t1)) # adapt to py3
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))  
    if args.mask_eval: # print for semantic mask evaluation
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(bg_abs_rel.mean(), bg_sq_rel.mean(), bg_rms.mean(), bg_log_rms.mean(), bg_d1_all.mean(), bg_a1.mean(), bg_a2.mean(), bg_a3.mean()))


main()