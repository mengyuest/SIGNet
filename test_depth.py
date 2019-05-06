from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from sig_model import *

import cv2


def test_depth(opt):
    ##### load testing list #####
    test_files_list_path = 'data/kitti/test_files_%s.txt' % opt.depth_test_split

    with open(test_files_list_path, 'r') as f:
        test_files = f.readlines()

        sem_test_dir=opt.sem_test_kitti_dir
        ins_test_dir=opt.ins_test_kitti_dir
        dataset_dir=opt.dataset_dir

        sem_test_files = [sem_test_dir + t[:-5]+".npy" for t in test_files]
        ins_test_files = [ins_test_dir + t[:-5]+"_instance_new.npy" for t in test_files]
        test_files = [dataset_dir + t[:-1] for t in test_files]
        

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    ##### init #####
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                    opt.img_height, opt.img_width, 3], name='raw_input')
    
    input_sem_uint8=None
    input_sem_map_uint8=None
    input_sem_mask_uint8=None
    input_sem_edge_uint8=None

    input_ins0_uint8=None
    input_ins0_map_uint8=None
    input_ins0_edge_uint8=None
    input_ins1_edge_uint8=None
    
    if opt.sem_assist and opt.sem_as_feat:
        if opt.one_hot_sem_feat:
            k=opt.sem_num_class
            input_sem_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                    opt.img_height, opt.img_width, k], name='raw_input_sem')
        else:
            if opt.sem_mask_feature:
                input_sem_mask_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                    opt.img_height, opt.img_width, 1], name='raw_input_sem_mask')
                
            elif opt.sem_edge_feature:
                input_sem_edge_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                    opt.img_height, opt.img_width, 1], name='raw_input_sem_edge')

            else:
                input_sem_map_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                    opt.img_height, opt.img_width, 1], name='raw_input_sem_map')
    if opt.ins_assist:
        k=opt.ins_num_class
        input_ins0_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
            opt.img_height, opt.img_width, k], name='raw_input_ins0')
        
        input_ins0_map_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
            opt.img_height, opt.img_width, 1], name='raw_input_ins0_map')
            
        if opt.ins0_edge_explore:
            input_ins0_edge_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                opt.img_height, opt.img_width, 1], name='raw_input_ins0_edge')

        if opt.ins1_edge_explore:
            input_ins1_edge_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                opt.img_height, opt.img_width, 1], name='raw_input_ins1_edge')
    
    # RUN MODEL NETWORK
    NONES=[None]*4
    model = SIGNetModel(opt, input_uint8, None, None, \
                [input_sem_uint8,  input_sem_map_uint8,  input_sem_mask_uint8,  input_sem_edge_uint8 ], NONES, \
                [input_ins0_uint8, input_ins0_map_uint8, input_ins0_edge_uint8, input_ins1_edge_uint8], NONES)


    fetches = { "depth": model.pred_depth[0] }

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    with tf.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), opt.batch_size):
            
            the_feed_dict={}

            inputs = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 3), dtype=np.uint8)

            inputs_sem = np.zeros((opt.batch_size, opt.img_height, opt.img_width, opt.sem_num_class),dtype=np.uint8)
            inputs_sem_map = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 1),dtype=np.uint8)
            inputs_sem_mask = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 1),dtype=np.uint8)
            inputs_sem_edge = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 1),dtype=np.uint8)

            inputs_ins0 = np.zeros((opt.batch_size, opt.img_height, opt.img_width, opt.ins_num_class),dtype=np.uint8)
            inputs_ins0_map = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 1),dtype=np.uint8)
            inputs_ins0_edge = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 1),dtype=np.uint8)
            inputs_ins1_edge = np.zeros((opt.batch_size, opt.img_height, opt.img_width, 1),dtype=np.uint8)

            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                
                fh = open(test_files[idx], 'rb') # adapt to py3 ref: https://github.com/python-pillow/Pillow/issues/1605
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                
                inputs[b] = np.array(scaled_im)

                # darking mode
                inputs[b] = inputs[b] * opt.lighting_factor

                #TODO 
                if opt.sem_assist and opt.sem_as_feat:
                    sem_np=np.load(sem_test_files[idx])
                    sem_im=pil.fromarray(sem_np)

                    scaled_sem_im = sem_im.resize((opt.img_width, opt.img_height), pil.NEAREST)
                    scaled_sem_im = np.array(scaled_sem_im) #TODO scaled_sem_im shape (128,416) 
                    scaled_sem_im_1c = np.expand_dims(scaled_sem_im, -1) #TODO scaled_sem_im_1c shape (128,416,1)

                    if opt.one_hot_sem_feat:
                        sem_oh_im = np.zeros(list(scaled_sem_im.shape)+[opt.sem_num_class])
                        sem_oh_im[list(np.indices(sem_oh_im.shape[:-1]))+[scaled_sem_im]]=1 #TODO one-hot encoding
                        inputs_sem[b] = np.array(sem_oh_im)
                    else:
                        if opt.sem_mask_feature:
                            if opt.sem_mask_pattern=="foreground":
                                inputs_sem[b] = (scaled_sem_im_1c>10).astype(np.uint8) 
                            else:
                                inputs_sem[b] = (scaled_sem_im_1c<=10).astype(np.uint8) 
                        elif opt.sem_edge_feature: #TODO scaled_sem_im_1c shape (128,416,1)
                            x = np.pad((scaled_sem_im_1c[:, :-1, :]!=scaled_sem_im_1c[:, 1:, :]).astype(np.uint8), ((0,0),(0,1),(0,0)), 'constant')
                            y = np.pad((scaled_sem_im_1c[:-1, :, :]!=scaled_sem_im_1c[1:, :, :]).astype(np.uint8), ((0,1),(0,0),(0,0)), 'constant')
                            inputs_sem[b] = x+y 
                        else:
                            inputs_sem[b] = scaled_sem_im_1c

                if opt.ins_assist:
                    ins_np=np.load(ins_test_files[idx]) # (375, 1242, 2)
                    ins0_im=pil.fromarray(ins_np[:,:,0])
                    ins1_im=pil.fromarray(ins_np[:,:,1])

                    scaled_ins0_im = ins0_im.resize((opt.img_width, opt.img_height), pil.NEAREST)
                    scaled_ins0_im = np.array(scaled_ins0_im) #TODO scaled_ins0_im shape (128,416)
                    scaled_ins0_im_1c = np.expand_dims(scaled_ins0_im, -1) #TODO scaled_ins0_im_1c shape (128,416,1)


                    scaled_ins1_im = ins1_im.resize((opt.img_width, opt.img_height), pil.NEAREST) 
                    scaled_ins1_im = np.array(scaled_ins1_im) #TODO scaled_ins0_im shape (128,416)
                    scaled_ins1_im_1c = np.expand_dims(scaled_ins1_im, -1) #TODO scaled_ins1_im_1c shape (128,416,1)

                    ins0_oh_im = np.zeros(list(scaled_ins0_im.shape)+[opt.ins_num_class])
                    ins0_oh_im[list(np.indices(ins0_oh_im.shape[:-1]))+[scaled_ins0_im]]=1 #TODO one-hot encoding
                    inputs_ins0[b] = np.array(ins0_oh_im)

                    inputs_ins0_map[b] = scaled_ins0_im_1c
                    
                    if opt.ins0_edge_explore:
                        x = np.pad((scaled_ins0_im_1c[:, :-1, :]!=scaled_ins0_im_1c[:, 1:, :]).astype(np.uint8), ((0,0),(0,1),(0,0)), 'constant')
                        y = np.pad((scaled_ins0_im_1c[:-1, :, :]!=scaled_ins0_im_1c[1:, :, :]).astype(np.uint8), ((0,1),(0,0),(0,0)), 'constant')
                        inputs_ins0_edge[b] = x+y 

                    if opt.ins1_edge_explore:
                        x = np.pad((scaled_ins1_im_1c[:, :-1, :]!=scaled_ins1_im_1c[:, 1:, :]).astype(np.uint8), ((0,0),(0,1),(0,0)), 'constant')
                        y = np.pad((scaled_ins1_im_1c[:-1, :, :]!=scaled_ins1_im_1c[1:, :, :]).astype(np.uint8), ((0,1),(0,0),(0,0)), 'constant')
                        inputs_ins1_edge[b] = x+y 

            # organize the feed_dict
            the_feed_dict[input_uint8]=inputs

            if opt.sem_assist and opt.sem_as_feat:
                if opt.one_hot_sem_feat:
                    the_feed_dict[input_sem_uint8]=inputs_sem
                else:
                    if opt.sem_mask_feature:
                        the_feed_dict[input_sem_mask_uint8]=inputs_sem_mask
                    elif opt.sem_edge_feature:
                        the_feed_dict[input_sem_edge_uint8]=inputs_sem_edge
                    else:
                        the_feed_dict[input_sem_map_uint8]=inputs_sem_map

            if opt.ins_assist:
                the_feed_dict[input_ins0_uint8]=inputs_ins0
                
                the_feed_dict[input_ins0_map_uint8]=inputs_ins0_map
                if opt.ins0_edge_explore:
                    the_feed_dict[input_ins0_edge_uint8]=inputs_ins0_edge
                if opt.ins1_edge_explore:
                    the_feed_dict[input_ins1_edge_uint8]=inputs_ins1_edge
               
            pred = sess.run(fetches, feed_dict=the_feed_dict)
            
            for b in range(opt.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b,:,:,0])

        np.save(opt.output_dir + '/' + 'model.npy', pred_all)