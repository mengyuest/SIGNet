from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sig_nets import *
from utils import *
#TODO randomness
import random

class SIGNetModel(object):

    def __init__(self, opt, tgt_image, src_image_stack, intrinsics,
                       tgt_sem_tuple=[None,None,None,None], src_sem_stack_tuple=[None,None,None,None],
                       tgt_ins_tuple=[None,None,None,None], src_ins_stack_tuple=[None,None,None,None]):
        self.opt = opt
        #TODO random
        seed = 8964
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.tgt_image = self.preprocess_image(tgt_image)
        self.src_image_stack = self.preprocess_image(src_image_stack)
        self.intrinsics = intrinsics

        #TODO add semantics feed-in
        tgt_sem, tgt_sem_map, tgt_sem_mask, tgt_sem_edge = tgt_sem_tuple
        src_sem_stack, src_sem_map_stack, src_sem_mask_stack, src_sem_edge_stack = src_sem_stack_tuple
        
        self.tgt_sem_map = self.preprocess_sem(tgt_sem_map, opt.sem_num_class-1, True)
        self.src_sem_map_stack = self.preprocess_sem(src_sem_map_stack, opt.sem_num_class-1, True)

        self.tgt_sem = self.preprocess_sem(tgt_sem)
        self.src_sem_stack = self.preprocess_sem(src_sem_stack)

        self.tgt_sem_mask = self.preprocess_sem(tgt_sem_mask)
        self.src_sem_mask_stack = self.preprocess_sem(src_sem_mask_stack)
        
        self.tgt_sem_edge = self.preprocess_sem(tgt_sem_edge)
        self.src_sem_edge_stack = self.preprocess_sem(src_sem_edge_stack)

        #TODO add instance feed-in (preproc)
        tgt_ins0, tgt_ins0_map, tgt_ins0_edge, tgt_ins1_edge = tgt_ins_tuple
        src_ins0_stack, src_ins0_map_stack, src_ins0_edge_stack, src_ins1_edge_stack = src_ins_stack_tuple
        
        self.tgt_ins0_map = self.preprocess_sem(tgt_ins0_map, opt.ins_num_class-1, True)
        self.src_ins0_map_stack = self.preprocess_sem(src_ins0_map_stack, opt.ins_num_class-1, True)

        self.tgt_ins0 = self.preprocess_sem(tgt_ins0)
        self.src_ins0_stack = self.preprocess_sem(src_ins0_stack)

        self.tgt_ins0_edge = self.preprocess_sem(tgt_ins0_edge)
        self.src_ins0_edge_stack = self.preprocess_sem(src_ins0_edge_stack)

        self.tgt_ins1_edge = self.preprocess_sem(tgt_ins1_edge)
        self.src_ins1_edge_stack = self.preprocess_sem(src_ins1_edge_stack)

        self.build_model()

        if not opt.mode in ['train_rigid', 'train_flow']:
            return

        self.build_losses()

    def build_model(self):
        opt = self.opt
        self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)
        self.tgt_image_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) \
                                      for img in self.tgt_image_pyramid]

        # src images concated along batch dimension
        if self.src_image_stack != None:
            self.src_image_concat = tf.concat([self.src_image_stack[:,:,:,3*i:3*(i+1)] \
                                    for i in range(opt.num_source)], axis=0)
            self.src_image_concat_pyramid = self.scale_pyramid(self.src_image_concat, opt.num_scales)

        #TODO build pyramids for semantic segmentations
        if opt.sem_as_loss:
            K=opt.sem_num_class
            self.tgt_sem_pyramid = self.sem_scale_pyramid(self.tgt_sem, opt.num_scales)
            self.tgt_sem_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) \
                                        for img in self.tgt_sem_pyramid]

            if opt.sem_mask_explore:
                self.tgt_sem_mask_pyramid = self.sem_scale_pyramid(self.tgt_sem_mask, opt.num_scales)
                self.tgt_sem_mask_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) \
                            for img in self.tgt_sem_mask_pyramid]

            
            if opt.sem_edge_explore:
                self.tgt_sem_edge_pyramid = self.sem_scale_pyramid(self.tgt_sem_edge, opt.num_scales)

            # src sem concated along batch dimension
            if self.src_sem_stack != None:
                self.src_sem_concat = tf.concat([self.src_sem_stack[:,:,:,K*i:K*(i+1)] \
                                        for i in range(opt.num_source)], axis=0)
                self.src_sem_concat_pyramid = self.sem_scale_pyramid(self.src_sem_concat, opt.num_scales)

                if opt.sem_mask_explore:
                    self.src_sem_mask_concat = tf.concat([self.src_sem_mask_stack[:,:,:,1*i:1*(i+1)] \
                                        for i in range(opt.num_source)], axis=0)
                    self.src_sem_mask_concat_pyramid = self.sem_scale_pyramid(self.src_sem_mask_concat, opt.num_scales)

                if opt.sem_edge_explore:
                    self.src_sem_edge_concat = tf.concat([self.src_sem_edge_stack[:,:,:,1*i:1*(i+1)] \
                                        for i in range(opt.num_source)], axis=0)
                    self.src_sem_edge_concat_pyramid = self.sem_scale_pyramid(self.src_sem_edge_concat, opt.num_scales)

        #TODO build pyramids for instance segmentations
        if opt.ins_as_loss:
            K=opt.ins_num_class
            self.tgt_ins0_pyramid = self.sem_scale_pyramid(self.tgt_ins0, opt.num_scales)
            self.tgt_ins0_map_pyramid=self.sem_scale_pyramid(self.tgt_ins0_map, opt.num_scales, True)

            if opt.ins1_edge_explore:
                self.tgt_ins1_edge_pyramid = self.sem_scale_pyramid(self.tgt_ins1_edge, opt.num_scales)
                for i,img in enumerate(self.tgt_ins1_edge_pyramid):
                    tf.summary.image("aprior_tgt_ins1_edge_pyramid" + str(i), self.tgt_ins1_edge_pyramid[i], max_outputs=opt.max_outputs)
            if self.src_ins0_stack!=None:
                self.src_ins0_concat = tf.concat([self.src_ins0_stack[:,:,:,K*i:K*(i+1)] \
                                        for i in range(opt.num_source)], axis=0)
                self.src_ins0_concat_pyramid = self.sem_scale_pyramid(self.src_ins0_concat, opt.num_scales)
                
                self.src_ins0_map_concat = tf.concat([self.src_ins0_map_stack[:,:,:,1*i:1*(i+1)] \
                                        for i in range(opt.num_source)], axis=0)
                self.src_ins0_map_concat_pyramid = self.sem_scale_pyramid(self.src_ins0_map_concat, opt.num_scales, True)

                if opt.ins1_edge_explore:
                    self.src_ins1_edge_concat = tf.concat([self.src_ins1_edge_stack[:,:,:,1*i:1*(i+1)] \
                                        for i in range(opt.num_source)], axis=0)
                    self.src_ins1_edge_concat_pyramid = self.sem_scale_pyramid(self.src_ins1_edge_concat, opt.num_scales)

                    for i,img in enumerate(self.src_ins1_edge_concat_pyramid):
                        tf.summary.image("aprior_src_ins1_edge_concat_pyramid" + str(i), self.src_ins1_edge_concat_pyramid[i], max_outputs=opt.max_outputs)

        if opt.add_dispnet:
            self.build_dispnet()

        if opt.add_posenet:
            self.build_posenet()

        if opt.add_dispnet and opt.add_posenet:
            self.build_rigid_flow_warping()

        if opt.sem_assist and opt.add_segnet:
            self.build_segnet()

        if opt.add_flownet:
            self.build_flownet()
            if opt.mode == 'train_flow':
                self.build_full_flow_warping()
                if opt.flow_consistency_weight > 0:
                    self.build_flow_consistency()

    def build_dispnet(self):
        opt = self.opt

        # build dispnet_inputs
        if opt.mode == 'test_depth':
            # for test_depth mode we only predict the depth of the target image
            self.dispnet_inputs = self.tgt_image
        else:
            # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            self.dispnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.dispnet_inputs = tf.concat([self.dispnet_inputs, self.src_image_stack[:,:,:,3*i:3*(i+1)]], axis=0)

        dispnet_inputs_extras=[]
        # augment in channel dim with semantic information
        if opt.sem_as_feat:
            if opt.one_hot_sem_feat:
                k=opt.sem_num_class

                dispnet_inputs_sem = self.tgt_sem
                if opt.mode != 'test_depth':
                    for i in range(opt.num_source):
                        dispnet_inputs_sem = tf.concat([dispnet_inputs_sem, self.src_sem_stack[:,:,:,k*i:k*(i+1)]], axis=0)
            else:
                if opt.sem_mask_feature:
                    dispnet_inputs_sem = self.tgt_sem_mask
                    if opt.mode != 'test_depth':
                        for i in range(opt.num_source):
                            dispnet_inputs_sem = tf.concat([dispnet_inputs_sem, self.src_sem_mask_stack[:,:,:,i:(i+1)]], axis=0)
                elif opt.sem_edge_feature:
                    dispnet_inputs_sem = self.tgt_sem_edge
                    if opt.mode != 'test_depth':
                        for i in range(opt.num_source):
                            dispnet_inputs_sem = tf.concat([dispnet_inputs_sem, self.src_sem_edge_stack[:,:,:,i:(i+1)]], axis=0)
                else:
                    dispnet_inputs_sem = self.tgt_sem_map
                    if opt.mode != 'test_depth':
                        for i in range(opt.num_source):
                            dispnet_inputs_sem = tf.concat([dispnet_inputs_sem, self.src_sem_map_stack[:,:,:,i:(i+1)]], axis=0)
            dispnet_inputs_extras.append(dispnet_inputs_sem)

        if opt.ins_as_feat:
            if opt.ins0_onehot_feature:
                k=opt.ins_num_class
                dispnet_inputs_ins0 = self.tgt_ins0
                if opt.mode != "test_depth":
                    for i in range(opt.num_source):
                        dispnet_inputs_ins0 = tf.concat([dispnet_inputs_ins0, self.src_ins0_stack[:,:,:,k*i:k*(i+1)]], axis=0)
                dispnet_inputs_extras.append(dispnet_inputs_ins0)

            if opt.ins0_dense_feature:
                dispnet_inputs_ins0_map = self.tgt_ins0_map
                if opt.mode != "test_depth":
                    for i in range(opt.num_source):
                        dispnet_inputs_ins0_map = tf.concat([dispnet_inputs_ins0_map, self.src_ins0_map_stack[:,:,:,i:(i+1)]], axis=0)
                dispnet_inputs_extras.append(dispnet_inputs_ins0_map)

            if opt.ins0_edge_feature:
                dispnet_inputs_ins0_edge = self.tgt_ins0_edge
                if opt.mode != "test_depth":
                    for i in range(opt.num_source):
                        dispnet_inputs_ins0_edge = tf.concat([dispnet_inputs_ins0_edge, self.src_ins0_edge_stack[:,:,:,i:(i+1)]], axis=0)
                dispnet_inputs_extras.append(dispnet_inputs_ins0_edge)

            if opt.ins1_edge_feature:
                dispnet_inputs_ins1_edge = self.tgt_ins1_edge
                if opt.mode != "test_depth":
                    for i in range(opt.num_source):
                        dispnet_inputs_ins1_edge = tf.concat([dispnet_inputs_ins1_edge, self.src_ins1_edge_stack[:,:,:,i:(i+1)]], axis=0)
                dispnet_inputs_extras.append(dispnet_inputs_ins1_edge)
 
        self.dispnet_inputs_extra=None
        if len(dispnet_inputs_extras)>0:
            self.dispnet_inputs_extra = tf.concat(dispnet_inputs_extras, axis=3)

        #TODO not blocked (no new network) and having extra; concat image and sem
        if opt.block_dispnet_sem==False and self.dispnet_inputs_extra!=None: 
            self.dispnet_inputs = tf.concat([self.dispnet_inputs, self.dispnet_inputs_extra], axis=3)

        if opt.block_dispnet_sem and opt.new_sem_dispnet and self.dispnet_inputs_extra!=None:
            self.pred_disp = disp_net(opt, self.dispnet_inputs, False) + disp_net(opt, self.dispnet_inputs_extra, True)
        else:
            self.pred_disp = disp_net(opt, self.dispnet_inputs, False)


        if opt.scale_normalize:
            # As proposed in https://arxiv.org/abs/1712.00175, this can 
            # bring improvement in depth estimation, but not included in our paper.
            self.pred_disp = [self.spatial_normalize(disp) for disp in self.pred_disp]

        self.pred_depth = [1./d for d in self.pred_disp]
        
        #TODO Add multi-scale depth maps to TF summary.
        for i in range(len(self.pred_depth)):
            tf.summary.image('pred_depth_' + str(i), self.pred_depth[i], max_outputs=opt.max_outputs)


    def build_posenet(self):
        opt = self.opt

        # build posenet_inputs
        self.posenet_inputs = tf.concat([self.tgt_image, self.src_image_stack], axis=3)
        
        posenet_inputs_extras=[]
        #TODO adding semantic as input
        if opt.sem_as_feat:
            if opt.one_hot_sem_feat:
                posenet_inputs_sem = tf.concat([self.tgt_sem, self.src_sem_stack], axis=3) #TODO problem!!! dimension cat upon?
            else:
                if opt.sem_mask_feature:
                    posenet_inputs_sem = tf.concat([self.tgt_sem_mask, self.src_sem_mask_stack], axis=3)
                elif opt.sem_edge_feature:
                    posenet_inputs_sem = tf.concat([self.tgt_sem_edge, self.src_sem_edge_stack], axis=3)
                else:
                    posenet_inputs_sem = tf.concat([self.tgt_sem_map, self.src_sem_map_stack], axis=3)
            
            posenet_inputs_extras.append(posenet_inputs_sem)

        #TODO adding instance as input
        if opt.ins_as_feat:
            if opt.ins0_onehot_feature:
                posenet_inputs_extras.append(tf.concat([self.tgt_ins0, self.src_ins0_stack], axis=3))
            if opt.ins0_dense_feature:
                posenet_inputs_extras.append(tf.concat([self.tgt_ins0_map, self.src_ins0_map_stack], axis=3))
            if opt.ins0_edge_feature:
                posenet_inputs_extras.append(tf.concat([self.tgt_ins0_edge, self.src_ins0_edge_stack], axis=3))
            if opt.ins1_edge_feature:
                posenet_inputs_extras.append(tf.concat([self.tgt_ins1_edge, self.src_ins1_edge_stack], axis=3))

        self.posenet_inputs_extra=None
        if len(posenet_inputs_extras)>0:
            self.posenet_inputs_extra = tf.concat(posenet_inputs_extras, axis=3)

        #TODO not blocked(no new network) and having extra; concat image and sem
        if opt.block_posenet_sem==False and self.posenet_inputs_extra!=None: 
            self.posenet_inputs = tf.concat([self.posenet_inputs, self.posenet_inputs_extra], axis=3)

        if opt.block_posenet_sem and opt.new_sem_posenet and self.posenet_inputs_extra!=None:
            self.pred_poses = pose_net(opt, self.posenet_inputs, False) + pose_net(opt, self.posenet_inputs_extra, True)
        else:
            self.pred_poses = pose_net(opt, self.posenet_inputs, False)


    #TODO build the simple transfer network for semantic segmentation
    def build_segnet(self):
        opt = self.opt

        # build segnet_inputs
        if opt.mode == 'test_depth':
            self.segnet_inputs = self.tgt_image
        else:
            self.segnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.segnet_inputs = tf.concat([self.segnet_inputs, self.src_image_stack[:,:,:,3*i:3*(i+1)]], axis=0)

        # concatenate disp prediction  N*W*C*4 (4=3+1)
        # N=batch_size*(num_source+1)
        self.segnet_inputs = tf.concat([self.segnet_inputs, self.pred_disp[0]], axis=3)

        # build segnet N*W*C*(channel_seg)
        # N=batch_size*(num_source+1)
        # channel_seg=19, 81, 19+81, 19+81+1...
        self.pred_seg = seg_net(opt, self.segnet_inputs)

        #TODO Add to TF summary.
        for i in range(len(self.pred_seg)):
            for k in range(self.pred_seg[i].shape[3]):
                tf.summary.image('pred_seg_%d_%d'%(i,k), self.pred_seg[i][:,:,:,k:k+1], max_outputs=opt.max_outputs)


    def build_rigid_flow_warping(self):
        opt = self.opt
        bs = opt.batch_size

        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []
        for s in range(opt.num_scales):
            for i in range(opt.num_source):
                fwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][:bs], axis=3),
                                 self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], False)
                bwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                                 self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], True)
                if not i:
                    fwd_rigid_flow_concat = fwd_rigid_flow
                    bwd_rigid_flow_concat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
                    bwd_rigid_flow_concat = tf.concat([bwd_rigid_flow_concat, bwd_rigid_flow], axis=0)
            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_concat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_concat)

        # warping by rigid flow
        self.fwd_rigid_warp_pyramid = [flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        self.bwd_rigid_warp_pyramid = [flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_rigid_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]

        #TODO Record forward rigid flow warping result on tensorboard
        for i in range(len(self.fwd_rigid_warp_pyramid)):
            tf.summary.image("fwd_rigid_warp_scale" + str(i), self.fwd_rigid_warp_pyramid[i], max_outputs=opt.max_outputs)
        #TODO Record backward rigid flow warping result on tensorboard
        for i in range(len(self.bwd_rigid_warp_pyramid)):
            tf.summary.image("bwd_rigid_warp_scale" + str(i), self.bwd_rigid_warp_pyramid[i], max_outputs=opt.max_outputs)


        # compute reconstruction error  
        self.fwd_rigid_error_pyramid = [self.image_similarity(self.fwd_rigid_warp_pyramid[s], self.tgt_image_tile_pyramid[s]) \
                                       for s in range(opt.num_scales)]      
        self.bwd_rigid_error_pyramid = [self.image_similarity(self.bwd_rigid_warp_pyramid[s], self.src_image_concat_pyramid[s]) \
                                       for s in range(opt.num_scales)]

        #TODO Record fwd rigid flow warp error on tensorboard
        self.fwd_rigid_error_scale=[]
        self.bwd_rigid_error_scale=[]
        for i in range(len(self.fwd_rigid_error_pyramid)):
            tmp_fwd_rigid_error_scale=tf.reduce_mean(self.fwd_rigid_error_pyramid[i],axis=3,keepdims=True)
            tf.summary.image("fwd_rigid_error_scale" + str(i), tmp_fwd_rigid_error_scale, max_outputs=opt.max_outputs)
            self.fwd_rigid_error_scale.append(tmp_fwd_rigid_error_scale)
        #TODO Record bwd rigid flow warp error on tensorboard
        for i in range(len(self.bwd_rigid_error_pyramid)):
            tmp_bwd_rigid_error_scale=tf.reduce_mean(self.bwd_rigid_error_pyramid[i],axis=3,keepdims=True)
            tf.summary.image("bwd_rigid_error_scale" + str(i), tmp_bwd_rigid_error_scale, max_outputs=opt.max_outputs)
            self.bwd_rigid_error_scale.append(tmp_bwd_rigid_error_scale)

        #TODO build rigid flow for semantic segmentations (similar to images)
        if opt.sem_as_loss:
            # semantic warping by rigid flow
            if opt.sem_warp_explore:
                self.fwd_sem_rigid_warp_pyramid = [sem_flow_warp(self.src_sem_concat_pyramid[s], self.fwd_rigid_flow_pyramid[s], opt.sem_nn_warp) \
                                            for s in range(opt.num_scales)]
                self.bwd_sem_rigid_warp_pyramid = [sem_flow_warp(self.tgt_sem_tile_pyramid[s], self.bwd_rigid_flow_pyramid[s], opt.sem_nn_warp) \
                                            for s in range(opt.num_scales)]

                # compute sem warp reconstruction error
                self.fwd_sem_rigid_warp_error_pyramid = [self.cal_sem_warp_error(self.fwd_sem_rigid_warp_pyramid[s], self.tgt_sem_tile_pyramid[s], s) \
                                                for s in range(opt.num_scales)]
                self.bwd_sem_rigid_warp_error_pyramid = [self.cal_sem_warp_error(self.bwd_sem_rigid_warp_pyramid[s], self.src_sem_concat_pyramid[s], s) \
                                                for s in range(opt.num_scales)]
                
                #TODO Record fwd rigid flow warp error on tensorboard
                for i in range(len(self.fwd_sem_rigid_warp_error_pyramid)):
                    tf.summary.image("fwd_sem_rigid_error_scale" + str(i), tf.reduce_mean(self.fwd_sem_rigid_warp_error_pyramid[i],axis=3,keepdims=True), max_outputs=opt.max_outputs)
                #TODO Record bwd rigid flow warp error on tensorboard
                for i in range(len(self.bwd_sem_rigid_warp_error_pyramid)):
                    tf.summary.image("bwd_sem_rigid_error_scale" + str(i), tf.reduce_mean(self.bwd_sem_rigid_warp_error_pyramid[i],axis=3,keepdims=True), max_outputs=opt.max_outputs)
            
            #TODO Use sem mask to find error on **warped images**
            if opt.sem_mask_explore:
                self.fwd_sem_mask_error_pyramid = [self.image_similarity(self.fwd_rigid_warp_pyramid[s], self.tgt_image_tile_pyramid[s], self.tgt_sem_mask_tile_pyramid[s]) \
                                                for s in range(opt.num_scales)]      
                self.bwd_sem_mask_error_pyramid = [self.image_similarity(self.bwd_rigid_warp_pyramid[s], self.src_image_concat_pyramid[s], self.src_sem_mask_concat_pyramid[s]) \
                                                for s in range(opt.num_scales)]
                self.fwd_sem_mask_error_scale=[]
                self.bwd_sem_mask_error_scale=[]

                #TODO Record fwd rigid flow warp error on tensorboard
                for i in range(len(self.fwd_sem_mask_error_pyramid)):
                    tmp_fwd_sem_mask_error_scale=tf.reduce_mean(self.fwd_sem_mask_error_pyramid[i],axis=3,keepdims=True)
                    tf.summary.image("fwd_sem_mask_error_scale" + str(i), tmp_fwd_sem_mask_error_scale, max_outputs=opt.max_outputs)
                    self.fwd_sem_mask_error_scale.append(tmp_fwd_sem_mask_error_scale)
                #TODO Record bwd rigid flow warp error on tensorboard
                for i in range(len(self.bwd_sem_mask_error_pyramid)):
                    tmp_bwd_sem_mask_error_scale=tf.reduce_mean(self.bwd_sem_mask_error_pyramid[i],axis=3,keepdims=True)
                    tf.summary.image("bwd_sem_mask_error_scale" + str(i), tmp_bwd_sem_mask_error_scale, max_outputs=opt.max_outputs)
                    self.bwd_sem_mask_error_scale.append(tmp_bwd_sem_mask_error_scale)

    def build_flownet(self):
        opt = self.opt

        # build flownet_inputs
        self.fwd_flownet_inputs = tf.concat([self.tgt_image_tile_pyramid[0], self.src_image_concat_pyramid[0]], axis=3)
        self.bwd_flownet_inputs = tf.concat([self.src_image_concat_pyramid[0], self.tgt_image_tile_pyramid[0]], axis=3)
        if opt.flownet_type == 'residual':
            self.fwd_flownet_inputs = tf.concat([self.fwd_flownet_inputs,
                                      self.fwd_rigid_warp_pyramid[0],
                                      self.fwd_rigid_flow_pyramid[0],
                                      self.L2_norm(self.fwd_rigid_error_pyramid[0])], axis=3)
            self.bwd_flownet_inputs = tf.concat([self.bwd_flownet_inputs,
                                      self.bwd_rigid_warp_pyramid[0],
                                      self.bwd_rigid_flow_pyramid[0],
                                      self.L2_norm(self.bwd_rigid_error_pyramid[0])], axis=3)
        self.flownet_inputs = tf.concat([self.fwd_flownet_inputs, self.bwd_flownet_inputs], axis=0)
        
        # build flownet
        self.pred_flow = flow_net(opt, self.flownet_inputs)

        # unnormalize pyramid flow back into pixel metric
        for s in range(opt.num_scales):
            curr_bs, curr_h, curr_w, _ = self.pred_flow[s].get_shape().as_list()
            scale_factor = tf.cast(tf.constant([curr_w, curr_h], shape=[1,1,1,2]), 'float32')
            scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
            self.pred_flow[s] = self.pred_flow[s] * scale_factor

        # split forward/backward flows
        self.fwd_full_flow_pyramid = [self.pred_flow[s][:opt.batch_size*opt.num_source] for s in range(opt.num_scales)]
        self.bwd_full_flow_pyramid = [self.pred_flow[s][opt.batch_size*opt.num_source:] for s in range(opt.num_scales)]

        # residual flow postprocessing
        if opt.flownet_type == 'residual':
            self.fwd_full_flow_pyramid = [self.fwd_full_flow_pyramid[s] + self.fwd_rigid_flow_pyramid[s] for s in range(opt.num_scales)]
            self.bwd_full_flow_pyramid = [self.bwd_full_flow_pyramid[s] + self.bwd_rigid_flow_pyramid[s] for s in range(opt.num_scales)]   

    def build_full_flow_warping(self):
        opt = self.opt
        
        # warping by full flow
        self.fwd_full_warp_pyramid = [flow_warp(self.src_image_concat_pyramid[s], self.fwd_full_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        self.bwd_full_warp_pyramid = [flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_full_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]

        # compute reconstruction error  
        self.fwd_full_error_pyramid = [self.image_similarity(self.fwd_full_warp_pyramid[s], self.tgt_image_tile_pyramid[s]) \
                                       for s in range(opt.num_scales)]      
        self.bwd_full_error_pyramid = [self.image_similarity(self.bwd_full_warp_pyramid[s], self.src_image_concat_pyramid[s]) \
                                       for s in range(opt.num_scales)]    

    def build_flow_consistency(self):
        opt = self.opt

        # warp pyramid full flow
        self.bwd2fwd_flow_pyramid = [flow_warp(self.bwd_full_flow_pyramid[s], self.fwd_full_flow_pyramid[s]) \
                                    for s in range(opt.num_scales)]
        self.fwd2bwd_flow_pyramid = [flow_warp(self.fwd_full_flow_pyramid[s], self.bwd_full_flow_pyramid[s]) \
                                    for s in range(opt.num_scales)]

        # calculate flow consistency
        self.fwd_flow_diff_pyramid = [tf.abs(self.bwd2fwd_flow_pyramid[s] + self.fwd_full_flow_pyramid[s]) for s in range(opt.num_scales)]
        self.bwd_flow_diff_pyramid = [tf.abs(self.fwd2bwd_flow_pyramid[s] + self.bwd_full_flow_pyramid[s]) for s in range(opt.num_scales)]

        # build flow consistency condition
        self.fwd_consist_bound = [opt.flow_consistency_beta * self.L2_norm(self.fwd_full_flow_pyramid[s]) * 2**s for s in range(opt.num_scales)]
        self.bwd_consist_bound = [opt.flow_consistency_beta * self.L2_norm(self.bwd_full_flow_pyramid[s]) * 2**s for s in range(opt.num_scales)]
        self.fwd_consist_bound = [tf.stop_gradient(tf.maximum(v, opt.flow_consistency_alpha)) for v in self.fwd_consist_bound]
        self.bwd_consist_bound = [tf.stop_gradient(tf.maximum(v, opt.flow_consistency_alpha)) for v in self.bwd_consist_bound]

        # build flow consistency mask
        self.noc_masks_src = [tf.cast(tf.less(self.L2_norm(self.bwd_flow_diff_pyramid[s]) * 2**s, 
                             self.bwd_consist_bound[s]), tf.float32) for s in range(opt.num_scales)]
        self.noc_masks_tgt = [tf.cast(tf.less(self.L2_norm(self.fwd_flow_diff_pyramid[s]) * 2**s,
                             self.fwd_consist_bound[s]), tf.float32) for s in range(opt.num_scales)]

    def build_losses(self):
        opt = self.opt
        bs = opt.batch_size
        rigid_warp_loss = 0.0
        disp_smooth_loss = 0.0

        #TODO sem loss
        sem_warp_loss=0.0
        sem_mask_loss=0.0
        sem_edge_loss=0.0
        
        #TODO sem guidance loss
        sem_seg_loss=0.0
        ins0_seg_loss=0.0
        ins1_edge_seg_loss=0.0

        flow_warp_loss = 0.0
        flow_smooth_loss = 0.0
        flow_consistency_loss = 0.0

        for s in range(opt.num_scales):
            # rigid_warp_loss
            if opt.mode == 'train_rigid' and opt.rigid_warp_weight > 0:
                rigid_warp_loss += opt.rigid_warp_weight*opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_rigid_error_pyramid[s]) + \
                                 tf.reduce_mean(self.bwd_rigid_error_pyramid[s]))
            #TODO sem_as_loss
            if opt.mode == 'train_rigid' and opt.sem_as_loss:

                if opt.sem_warp_explore:
                    sem_warp_loss += opt.sem_warp_weight * opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_sem_rigid_warp_error_pyramid[s]) + tf.reduce_mean(self.bwd_sem_rigid_warp_error_pyramid[s]))
                
                if opt.sem_mask_explore:
                    sem_mask_loss += opt.sem_mask_weight * opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_sem_mask_error_pyramid[s]) + \
                                 tf.reduce_mean(self.bwd_sem_mask_error_pyramid[s]))
                
                if opt.sem_edge_explore:
                    sem_edge_loss += opt.sem_edge_weight /(2**s) * self.compute_sem_edge_smooth_loss(self.pred_disp[s],
                                tf.concat([self.tgt_sem_edge_pyramid[s], self.src_sem_edge_concat_pyramid[s]], axis=0))            

            # disp_smooth_loss
            if opt.mode == 'train_rigid' and opt.disp_smooth_weight > 0:
                disp_smooth_loss += opt.disp_smooth_weight/(2**s) * self.compute_smooth_loss(self.pred_disp[s],
                                tf.concat([self.tgt_image_pyramid[s], self.src_image_concat_pyramid[s]], axis=0))

            # flow_warp_loss
            if opt.mode == 'train_flow' and opt.flow_warp_weight > 0:
                if opt.flow_consistency_weight == 0:
                    flow_warp_loss += opt.flow_warp_weight*opt.num_source/2 * \
                                (tf.reduce_mean(self.fwd_full_error_pyramid[s]) + tf.reduce_mean(self.bwd_full_error_pyramid[s]))
                else:
                    flow_warp_loss += opt.flow_warp_weight*opt.num_source/2 * \
                                (tf.reduce_sum(tf.reduce_mean(self.fwd_full_error_pyramid[s], axis=3, keepdims=True) * \
                                 self.noc_masks_tgt[s]) / tf.reduce_sum(self.noc_masks_tgt[s]) + \
                                 tf.reduce_sum(tf.reduce_mean(self.bwd_full_error_pyramid[s], axis=3, keepdims=True) * \
                                 self.noc_masks_src[s]) / tf.reduce_sum(self.noc_masks_src[s]))

            # flow_smooth_loss
            if opt.mode == 'train_flow' and opt.flow_smooth_weight > 0:
                flow_smooth_loss += opt.flow_smooth_weight/(2**(s+1)) * \
                                (self.compute_flow_smooth_loss(self.fwd_full_flow_pyramid[s], self.tgt_image_tile_pyramid[s]) +
                                self.compute_flow_smooth_loss(self.bwd_full_flow_pyramid[s], self.src_image_concat_pyramid[s]))

            # flow_consistency_loss
            if opt.mode == 'train_flow' and opt.flow_consistency_weight > 0:
                flow_consistency_loss += opt.flow_consistency_weight/2 * \
                                (tf.reduce_sum(tf.reduce_mean(self.fwd_flow_diff_pyramid[s] , axis=3, keepdims=True) * \
                                 self.noc_masks_tgt[s]) / tf.reduce_sum(self.noc_masks_tgt[s]) + \
                                 tf.reduce_sum(tf.reduce_mean(self.bwd_flow_diff_pyramid[s] , axis=3, keepdims=True) * \
                                 self.noc_masks_src[s]) / tf.reduce_sum(self.noc_masks_src[s]))

            #TODO segmentation guidance loss(single-scale, cross-entropy)
            #     order shall be sem|ins0|ins1_edge, each of them can be omitted
            #     sem 0-18 (1 time)
            #     ins0 19-99, or 0-80 (2 times)
            #     ins1_edge 0-0, 81-81 or 100-100 (4 times)
            if opt.mode=='train_rigid' and opt.sem_assist and opt.add_segnet:
                n_sem=opt.sem_num_class
                n_ins=opt.ins_num_class
                if opt.transfer_learn_sem:
                    sem_seg_loss += opt.sem_seg_weight/(2**(s+1)) * \
                                tf.reduce_mean(self.compute_cross_entropy(self.tgt_sem_pyramid[s][0], self.pred_seg[s][0,:,:,:n_sem])+ \
                                self.compute_cross_entropy(self.src_sem_concat_pyramid[s][0], self.pred_seg[s][1,:,:,:n_sem])+ \
                                self.compute_cross_entropy(self.src_sem_concat_pyramid[s][1], self.pred_seg[s][2,:,:,:n_sem]))

                    if opt.transfer_learn_ins0:
                        ins0_seg_loss += opt.ins0_seg_weight/(2**(s+1)) * \
                                tf.reduce_mean(self.compute_cross_entropy(self.tgt_ins0_pyramid[s][0], self.pred_seg[s][0,:,:,n_sem:n_sem+n_ins])+ \
                                self.compute_cross_entropy(self.src_ins0_concat_pyramid[s][0], self.pred_seg[s][1,:,:,n_sem:n_sem+n_ins])+ \
                                self.compute_cross_entropy(self.src_ins0_concat_pyramid[s][1], self.pred_seg[s][2,:,:,n_sem:n_sem+n_ins]))
                        if opt.transfer_learn_ins1_edge:
                            ins1_edge_seg_loss += opt.ins1_edge_seg_weight/(2**(s+1)) *\
                                tf.reduce_mean(self.L2_norm(self.tgt_ins1_edge_pyramid[s][0]/2-self.pred_seg[s][0,:,:,n_sem+n_ins:n_sem+n_ins+1],None, False)+ \
                                self.L2_norm(self.src_ins1_edge_concat_pyramid[s][0]/2-self.pred_seg[s][1,:,:,n_sem+n_ins:n_sem+n_ins+1],None, False)+ \
                                self.L2_norm(self.src_ins1_edge_concat_pyramid[s][1]/2-self.pred_seg[s][2,:,:,n_sem+n_ins:n_sem+n_ins+1],None, False))
                    elif opt.transfer_learn_ins1_edge:
                        ins1_edge_seg_loss += opt.ins1_edge_seg_weight/(2**(s+1))  * \
                            tf.reduce_mean(self.L2_norm(self.tgt_ins1_edge_pyramid[s][0]/2-self.pred_seg[s][0,:,:,n_sem: n_sem+1],None, False)+ \
                            self.L2_norm(self.src_ins1_edge_concat_pyramid[s][0]/2-self.pred_seg[s][1,:,:,n_sem: n_sem+1],None, False)+ \
                            self.L2_norm(self.src_ins1_edge_concat_pyramid[s][1]/2-self.pred_seg[s][2,:,:,n_sem: n_sem+1],None, False))
                
                else:
                    if opt.transfer_learn_ins0:
                        ins0_seg_loss += opt.ins0_seg_weight/(2**(s+1)) * \
                                tf.reduce_mean(self.compute_cross_entropy(self.tgt_ins0_pyramid[s][0], self.pred_seg[s][0,:,:,: n_ins])+ \
                                self.compute_cross_entropy(self.src_ins0_concat_pyramid[s][0], self.pred_seg[s][1,:,:,: n_ins])+ \
                                self.compute_cross_entropy(self.src_ins0_concat_pyramid[s][1], self.pred_seg[s][2,:,:,: n_ins]))
                        if opt.transfer_learn_ins1_edge:
                            ins1_edge_seg_loss += opt.ins1_edge_seg_weight/(2**(s+1))  * \
                                tf.reduce_mean(self.L2_norm(self.tgt_ins1_edge_pyramid[s][0]/2-self.pred_seg[s][0,:,:,n_ins: n_ins+1],None, False)+ \
                                self.L2_norm(self.src_ins1_edge_concat_pyramid[s][0]/2-self.pred_seg[s][1,:,:,n_ins: n_ins+1],None, False)+ \
                                self.L2_norm(self.src_ins1_edge_concat_pyramid[s][1]/2-self.pred_seg[s][2,:,:,n_ins: n_ins+1],None, False))
                    elif opt.transfer_learn_ins1_edge:
                        ins1_edge_seg_loss += opt.ins1_edge_seg_weight/(2**(s+1))  * \
                            tf.reduce_mean(self.L2_norm(self.tgt_ins1_edge_pyramid[s][0]/2-self.pred_seg[s][0,:,:,: 1],None, False)+ \
                            self.L2_norm(self.src_ins1_edge_concat_pyramid[s][0]/2-self.pred_seg[s][1,:,:,: 1],None, False)+ \
                            self.L2_norm(self.src_ins1_edge_concat_pyramid[s][1]/2-self.pred_seg[s][2,:,:,: 1],None, False))

        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.total_loss = 0.0
        if opt.use_regularization:
            self.total_loss += regularization_loss
        self.img_loss = 0.0
        self.rigid_warp_loss = 0.0
        self.disp_smooth_loss = 0.0

        if opt.sem_as_loss:
            self.sem_loss = 0.0
            if opt.sem_warp_explore:
                self.sem_warp_loss=0.0
            if opt.sem_mask_explore:
                self.sem_mask_loss=0.0
            if opt.sem_edge_explore:
                self.sem_edge_loss=0.0
        
        if opt.sem_assist and opt.add_segnet:
            self.sem_seg_loss=0.0
            self.ins0_seg_loss=0.0
            self.ins1_edge_seg_loss=0.0

        if opt.ins_as_loss:
            self.ins_loss = 0.0

        #TODO modified loss function
        if opt.mode == 'train_rigid':
            self.rigid_warp_loss +=  rigid_warp_loss
            self.disp_smooth_loss += disp_smooth_loss
            self.img_loss = rigid_warp_loss + disp_smooth_loss
            self.total_loss += self.img_loss

            #TODO our sem loss
            if opt.sem_as_loss:
                if opt.sem_warp_explore:
                    self.sem_warp_loss = sem_warp_loss
                    self.sem_loss += self.sem_warp_loss
                if opt.sem_mask_explore:
                    self.sem_mask_loss = sem_mask_loss
                    self.sem_loss += self.sem_mask_loss 
                if opt.sem_edge_explore:
                    self.sem_edge_loss = sem_edge_loss
                    self.sem_loss += self.sem_edge_loss
                self.total_loss += self.sem_loss
                
            if opt.sem_assist and opt.add_segnet:
                self.sem_seg_loss += sem_seg_loss
                self.ins0_seg_loss += ins0_seg_loss
                self.ins1_edge_seg_loss += ins1_edge_seg_loss
                self.total_loss += self.sem_seg_loss + self.ins0_seg_loss + self.ins1_edge_seg_loss

        if opt.mode == 'train_flow':
            self.total_loss += flow_warp_loss + flow_smooth_loss + flow_consistency_loss

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity(self, x, y, mask=None):
        #TODO here our mask has one channel, and img has 3 channels
        if mask!=None:
            x=x*tf.tile(mask,[1,1,1,3])
            y=y*tf.tile(mask,[1,1,1,3])
        return self.opt.alpha_recon_image * self.SSIM(x, y) + (1-self.opt.alpha_recon_image) * tf.abs(x-y)

    def L2_norm(self, x, axis=3, keepdims=True):
        curr_offset = 1e-10
        l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keepdims=keepdims)
        return l2_norm

    def compute_cross_entropy(self, labels, logits):
        cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        return cross_ent

    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keepdims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean

    def scale_pyramid(self, img, num_scales):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = int(h / ratio)
                nw = int(w / ratio)
                scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            return scaled_imgs

    def sem_scale_pyramid(self, img, num_scales, nearest_neighbor=False):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = int(h / ratio)
                nw = int(w / ratio)
                if nearest_neighbor:
                    scaled_imgs.append(tf.image.resize_nearest_neighbor(img, [nh, nw]))
                else:
                    scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            return scaled_imgs

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def compute_smooth_loss(self, disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

    def compute_sem_edge_smooth_loss(self, disp, sem_edge_img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)
        
        image_gradients_x = (sem_edge_img[:,:,:-1,:] + sem_edge_img[:,:,1:,:])/2
        image_gradients_y = (sem_edge_img[:,:-1,:,:] + sem_edge_img[:,1:,:,:])/2 

        weights_x = tf.exp(-1*tf.abs(image_gradients_x))
        weights_y = tf.exp(-1*tf.abs(image_gradients_y))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(tf.expand_dims(flow[:,:,:,i], -1), img)
        return smoothness/2

    def cal_sem_warp_error(self, pred_img, gt_img, scale):
        return self.L2_norm(pred_img-gt_img) * 2**scale

    def preprocess_image(self, image):
        # Assuming input image is uint8
        if image == None:
            return None
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image * 2. -1.

    #TODO preprocessing for semantic parts
    def preprocess_sem(self, sem, max_value=1.0, normalize=False):
        # Assuming input image is uint8
        if sem == None:
            return None
        else:
            sem= tf.cast(sem, dtype=tf.float32) / max_value
            if normalize:
                return sem * 2 - 1
            else:
                return sem
            
    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)