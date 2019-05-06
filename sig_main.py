from __future__ import division
import os
import time
import random
import pprint
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from sig_model import *
from test_depth import *
from test_pose import *
from test_flow import *
from data_loader import DataLoader

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string("mode",                         "",    "(train_rigid, train_flow) or (test_depth, test_pose, test_flow)")
flags.DEFINE_string("dataset_dir",                  "",    "Dataset directory")
flags.DEFINE_string("init_ckpt_file",             None,    "Specific checkpoint file to initialize from")
flags.DEFINE_integer("batch_size",                   4,    "The size of of a sample batch")
flags.DEFINE_integer("num_threads",                 32,    "Number of threads for data loading")
flags.DEFINE_integer("img_height",                 128,    "Image height")
flags.DEFINE_integer("img_width",                  416,    "Image width")
flags.DEFINE_integer("seq_length",                   3,    "Sequence length for each example")

#TODO set log level
tf.logging.set_verbosity(tf.logging.WARN)

#TODO multi-gpu settings
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs")

#TODO S3 directory compatible setting
flags.DEFINE_string("filelist_dir", "", "File list directory")

#TODO low-level network flags to give more flexibility for training
flags.DEFINE_boolean("enable_batch_norm",        True,    "Shall we enable batch_norm configs")
flags.DEFINE_boolean("batch_norm_is_training",   True,    "Set the flag for slim.batch_norm is_training")

#TODO debug flags to help debugging
flags.DEFINE_boolean("train_lite",               False,    "Train on light weight dataset") # for quick debug
flags.DEFINE_integer("print_interval",             100,    "Interval for printing things out") # for debugging

#TODO tensorboard visualization
flags.DEFINE_string("summary_dir",                 "",    "Directory name to save the summaries")
flags.DEFINE_integer("save_summ_freq",            100,    "Save the TF summary every save_summ_freq iterations")
flags.DEFINE_integer("max_outputs",                 4,    "How many images per mini-batch we want to save") # see https://www.tensorflow.org/api_docs/python/tf/summary/image

#TODO regularization
flags.DEFINE_boolean("use_regularization",       False,   "Whether or not to use regularization term")

#TODO semantic-related flags to help depth estimation
flags.DEFINE_boolean("sem_assist",               False,    "Add semantic into account")
flags.DEFINE_boolean("load_from_raw",             True,     "Load from raw binary format")
flags.DEFINE_integer("sem_num_class",               19,    "Num of semantic classes")
flags.DEFINE_boolean("sem_as_loss",              False,    "Shall use semantic loss in loss function")
flags.DEFINE_boolean("sem_as_feat",              False,    "Shall add semantic as a feature in depth net")
flags.DEFINE_boolean("one_hot_sem_feat",         True,    "Shall we use one hot version of sem as input")
flags.DEFINE_float("sem_feat_weight",               .0,    "The weight of semantic feature in the network")
flags.DEFINE_boolean("fixed_posenet",            False,    "Shall we fixed the weight for posenet?")
flags.DEFINE_string("sem_test_kitti_dir",           "",    "Dir for semantic kitti testset")
flags.DEFINE_boolean("sem_nn_pyramid",           False,    "Use nearest neighbor to generate sem pyramid")
flags.DEFINE_boolean("sem_nn_warp",              False,    "Use nearest neighbor to execute flow warp")

#TODO explorations about the semantic loss: `sem_as_loss`==True
flags.DEFINE_boolean("sem_warp_explore",         False,    "Warp the semantic and compute the loss")
flags.DEFINE_string("sem_warp_function",            "",    "Loss function for semantic warping loss") 
flags.DEFINE_float("sem_warp_weight",              0.0,    "The weight for semantic warping loss") 
flags.DEFINE_boolean("sem_mask_explore",         False,    "Extact the mask and guide the img warp loss")
flags.DEFINE_boolean("sem_mask_feature",         False,    "Extact the mask and use as sem features, sem_mask_explore=True")
flags.DEFINE_string("sem_mask_pattern",             "",    "Patterns to extract the mask")
flags.DEFINE_string("sem_mask_function",            "",    "Functions for the guidance to the img warp loss")
flags.DEFINE_float("sem_mask_weight",              0.0,    "The weight for this img warp loss")
flags.DEFINE_boolean("sem_edge_explore",         False,    "Extact the edge and guide the smooth term")
flags.DEFINE_boolean("sem_edge_feature",         False,    "Extact the edge and use as sem features, sem_edge_explore=True")
flags.DEFINE_string("sem_edge_pattern",             "",    "Patterns to extract the edge")
flags.DEFINE_string("sem_edge_function",            "",    "Functions for the guidance to the smooth term")
flags.DEFINE_float("sem_edge_weight",              0.0,    "The weight for this smooth term")
flags.DEFINE_boolean("data_aug_cast",             True,    "adapt to previous version, but try to avoid this in future version")

flags.DEFINE_float("lighting_factor",         1.0,       "lighting factor")
flags.DEFINE_boolean("use_sem_weight_decay",    False,     "Adapt sem weight decay for sem features among training")
flags.DEFINE_string("sem_feat_struct",      "channel",     "Sem network type: channel or branch wise")
flags.DEFINE_float("weight_init_value",           1.0,     "Initial value for sem features. May 1.0 for type-I and 0.5 for type-II")
flags.DEFINE_string("weight_decay_rule",           "",     "1) Finally comes to zero 2) at some drop speed defined")

#TODO explorations about semantic-guided depth prediction network
flags.DEFINE_boolean("add_segnet",              False,     "To enable the transfer network to learn multi scale semantic segmentation from depth")
flags.DEFINE_string("transfer_network_structure",  "",     "The structure for that transfer network")
flags.DEFINE_float("sem_seg_weight",              0.0,     "The weight for semantic segmentation guidance loss weight")
flags.DEFINE_float("ins0_seg_weight",             0.0,     "The weight for instance class segmentation guidance loss weight")
flags.DEFINE_float("ins1_edge_seg_weight",        0.0,     "The weight for instance id edge segmentation guidance loss weight")
flags.DEFINE_boolean("transfer_learn_sem",         True,      "Use transfer network to learn semantic")
flags.DEFINE_boolean("transfer_learn_ins0",       False,      "Use transfer network to learn ins0")
flags.DEFINE_boolean("transfer_learn_ins1_edge",  False,      "Use transfer network to learn ins1 edges")

#TODO  Try to observe the variation of architectures handling semantic inputs
flags.DEFINE_boolean("block_dispnet_sem",       False,     "Not to receive the semantic input for dispnet")
flags.DEFINE_boolean("block_posenet_sem",       False,     "Not to receive the semantic input for posenet")
flags.DEFINE_boolean("new_sem_dispnet",         False,     "To build a new dispnet for receiving semantic input")
flags.DEFINE_boolean("new_sem_posenet",         False,     "To build a new posenet for receiving semantic input")

#TODO instance-related flags to help depth estimation
flags.DEFINE_boolean("ins_assist",               False,    "Add instance into account")
flags.DEFINE_integer("ins_num_class",               81,     "Maximum number of instances labels")
flags.DEFINE_boolean("ins_as_loss",              False,    "Shall use instance loss in loss function")
flags.DEFINE_float("ins_l2_norm_weight",           .3,     "The weight of ins l2-norm in loss function")
flags.DEFINE_boolean("ins_as_feat",              False,    "Shall add instance as a feature in depth net")

flags.DEFINE_boolean("ins0_dense_feature",         False,     "Shall use instance channel 0 dense info as features?")
flags.DEFINE_boolean("ins0_onehot_feature",         False,     "Shall we use one hot version of ins as input")
flags.DEFINE_boolean("ins0_edge_explore",         False,     "Shall explore instance channel 0 edge info?")
flags.DEFINE_boolean("ins0_edge_feature",         False,     "Shall use instance channel 0 edge info as features?")
flags.DEFINE_boolean("ins1_dense_feature",         False,     "Shall use instance channel 1 dense info as features?")
flags.DEFINE_boolean("ins1_onehot_feature",         False,     "Shall we use one hot version of ins as input")
flags.DEFINE_boolean("ins1_edge_explore",         False,     "Shall explore instance channel 1 edge info?")
flags.DEFINE_boolean("ins1_edge_feature",         False,     "Shall use instance channel 1 edge info as features?")

flags.DEFINE_string("ins_train_kitti_dir",          "../../data/kitti_eigen_new_instance_labels_acc_only/", "kitt inst train path?")
flags.DEFINE_string("ins_test_kitti_dir",           "../../data/test_files_eigen_semantic_acc_only/",     "kitt inst testpath?")

flags.DEFINE_float("ins_feat_weight",               .0,    "The weight of instance feature in the network")
flags.DEFINE_boolean("ins_nn_pyramid",           False,    "Use nearest neighbor to generate ins pyramid")
flags.DEFINE_boolean("ins_nn_warp",              False,    "Use nearest neighbor to execute flow warp")

##### Training Configurations #####
flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate",             0.0002,    "Learning rate for adam")
flags.DEFINE_integer("max_to_keep",               5000,    "Maximum number of checkpoints to save")
flags.DEFINE_integer("max_steps",               300000,    "Maximum number of training iterations")
flags.DEFINE_integer("save_ckpt_freq",            5000,    "Save the checkpoint model every save_ckpt_freq iterations")
flags.DEFINE_float("alpha_recon_image",           0.85,    "Alpha weight between SSIM and L1 in reconstruction loss")

##### Configurations about DepthNet & PoseNet of SigNet #####
flags.DEFINE_string("dispnet_encoder",      "resnet50",    "Type of encoder for dispnet, vgg or resnet50")
flags.DEFINE_boolean("scale_normalize",          False,    "Spatially normalize depth prediction")
flags.DEFINE_float("rigid_warp_weight",            1.0,    "Weight for warping by rigid flow")
flags.DEFINE_float("disp_smooth_weight",           0.5,    "Weight for disp smoothness")

##### Configurations about ResFlowNet of SigNet (or DirFlowNetS) #####
flags.DEFINE_string("flownet_type",         "residual",    "type of flownet, residual or direct")
flags.DEFINE_float("flow_warp_weight",             1.0,    "Weight for warping by full flow")
flags.DEFINE_float("flow_smooth_weight",           0.2,    "Weight for flow smoothness")
flags.DEFINE_float("flow_consistency_weight",      0.2,    "Weight for bidirectional flow consistency")
flags.DEFINE_float("flow_consistency_alpha",       3.0,    "Alpha for flow consistency check")
flags.DEFINE_float("flow_consistency_beta",       0.05,    "Beta for flow consistency check")

##### Testing Configurations #####
flags.DEFINE_string("output_dir",                 None,    "Test result output directory")
flags.DEFINE_string("depth_test_split",        "eigen",    "KITTI depth split, eigen or stereo")
flags.DEFINE_integer("pose_test_seq",                9,    "KITTI Odometry Sequence ID to test")

##### adapt to tf-1.10 +py3 ref: https://github.com/tinghuiz/SfMLearner/pull/70/commits/ec3007d82a7d2205ec5e5ffb5fc99729d31faf88
flags.DEFINE_integer("num_source",                   2, "Number of source images")
flags.DEFINE_integer("num_scales",                   4, "Number of scaling points")
flags.DEFINE_boolean("add_flownet",              False, "")
flags.DEFINE_boolean("add_dispnet",              False, "")
flags.DEFINE_boolean("add_posenet",              False, "")
opt = flags.FLAGS

def train():

    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0,
                                name='global_step',
                                trainable=False)
        incr_global_step = tf.assign(global_step,
                                     global_step+1)
        optim = tf.train.AdamOptimizer(opt.learning_rate, 0.9)

        loader = DataLoader(opt)

        losses=[]
        
        img_losses=[]
        rigid_warp_losses=[]
        disp_smooth_losses=[]
        
        sem_losses=[]
        sem_warp_losses=[]
        sem_mask_losses=[]
        sem_edge_losses=[]

        sem_seg_losses=[]
        ins0_seg_losses=[]
        ins1_edge_seg_losses=[]
        
        ins_losses=[]

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(opt.num_gpus):
                with tf.device('/gpu:{:d}'.format(i)):
                    with tf.name_scope('gpu{:d}'.format(i)):
                        # Get images batch from data loader
                        tgt_image, src_image_stack, intrinsics, tgt_sem_tuple, src_sem_stack_tuple, tgt_ins_tuple, src_ins_stack_tuple = loader.load_train_batch()      

                        # Build Model
                        model = SIGNetModel(opt, tgt_image, src_image_stack, intrinsics, tgt_sem_tuple, src_sem_stack_tuple, tgt_ins_tuple, src_ins_stack_tuple)
                        
                        # Handle losses
                        losses.append(model.total_loss)
                        tf.get_variable_scope().reuse_variables()

                        img_losses.append(model.img_loss)
                        rigid_warp_losses.append(model.rigid_warp_loss)
                        disp_smooth_losses.append(model.disp_smooth_loss)
                        if opt.sem_as_loss:
                            sem_losses.append(model.sem_loss)
                            if opt.sem_warp_explore:
                                sem_warp_losses.append(model.sem_warp_loss)
                            if opt.sem_mask_explore:
                                sem_mask_losses.append(model.sem_mask_loss)
                            if opt.sem_edge_explore:
                                sem_edge_losses.append(model.sem_edge_loss)
                        if opt.ins_as_loss:
                            ins_losses.append(model.ins_loss)          
                                      
                        if opt.sem_assist and opt.add_segnet:
                            sem_seg_losses.append(model.sem_seg_loss)
                            ins0_seg_losses.append(model.ins0_seg_loss)
                            ins1_edge_seg_losses.append(model.ins1_edge_seg_loss)

                        #TODO tensorboard
                        tf.summary.image('tgt_image_g%02d'%(i), tgt_image, max_outputs=opt.max_outputs)
                        tf.summary.image('src_image_prev_g%02d'%(i), src_image_stack[:, :, :, :3], max_outputs=opt.max_outputs)
                        tf.summary.image('src_image_next_g%02d'%(i), src_image_stack[:, :, :, 3:], max_outputs=opt.max_outputs)
                        tf.summary.scalar('loss_g%02d'%(i), model.total_loss)
                        tf.summary.scalar('img_loss_g%02d'%(i), model.img_loss)
                        tf.summary.scalar('rigid_warp_loss_g%02d'%(i),model.rigid_warp_loss)
                        tf.summary.scalar('disp_smooth_loss_g%02d'%(i),model.disp_smooth_loss)

                        if opt.sem_as_loss:
                            tf.summary.scalar('sem_loss_g%02d'%(i), model.sem_loss)
                            if opt.sem_warp_explore:
                                tf.summary.scalar('sem_warp_loss_g%02d'%(i), model.sem_warp_loss)
                        if opt.ins_as_loss:
                            tf.summary.scalar('ins_loss_g%02d'%(i), model.ins_loss)
                        
                        if opt.sem_assist and opt.add_segnet:
                            tf.summary.scalar('sem_seg_loss_g%02d'%(i), model.sem_seg_loss)
                            tf.summary.scalar('ins0_seg_loss_g%02d'%(i), model.ins0_seg_loss)
                            tf.summary.scalar('ins1_edge_seg_loss_g%02d'%(i), model.ins1_edge_seg_loss)
                        
                        #TODO Add bookkeeping ops
                        if i==0:
                            # Train Op
                            if opt.mode == 'train_flow' and opt.flownet_type == "residual":
                                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "flow_net")
                            else:
                                #TODO try to enable a solution to fix posenet weight in first stage
                                if opt.mode == 'train_rigid' and opt.fixed_posenet:
                                    if opt.new_sem_dispnet:
                                        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "depth_sem_net")
                                    else:    
                                        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "depth_net")
                                else:
                                    train_vars = [var for var in tf.trainable_variables()]

                            loading_net=["depth_net", "pose_net"]

                            if opt.new_sem_dispnet:
                                loading_net.append("depth_sem_net")
                            if opt.new_sem_posenet:
                                loading_net.append("pose_sem_net")

                            vars_to_restore = slim.get_variables_to_restore(include=loading_net)

                            if opt.init_ckpt_file != None:
                                init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                                                                opt.init_ckpt_file, vars_to_restore)

        #TODO Cal mean losses among gpus, and track the loss in TF Summary.
        loss = tf.stack(axis=0, values=losses)
        loss = tf.reduce_mean(loss, 0)
        tf.summary.scalar('loss', loss)

        rigid_warp_loss = tf.stack(axis=0, values=rigid_warp_losses)
        rigid_warp_loss = tf.reduce_mean(rigid_warp_loss, 0)
        tf.summary.scalar('rigid_warp_loss', rigid_warp_loss)
        tf.summary.scalar('unit_rigid_warp_loss', rigid_warp_loss/ (opt.rigid_warp_weight+tf.convert_to_tensor(1e-8, dtype=tf.float32)
) )

        disp_smooth_loss = tf.stack(axis=0, values=disp_smooth_losses)
        disp_smooth_loss = tf.reduce_mean(disp_smooth_loss, 0)
        tf.summary.scalar('disp_smooth_loss', disp_smooth_loss)
        tf.summary.scalar('unit_disp_smooth_loss', disp_smooth_loss/ (opt.disp_smooth_weight+tf.convert_to_tensor(1e-8, dtype=tf.float32)
) )

        img_loss = tf.stack(axis=0, values=img_losses)
        img_loss = tf.reduce_mean(img_loss, 0)
        tf.summary.scalar('img_loss', img_loss)

        if opt.sem_as_loss:
            sem_loss = tf.stack(axis=0, values=sem_losses)
            sem_loss = tf.reduce_mean(sem_loss, 0)
            tf.summary.scalar('sem_loss', sem_loss)

            if opt.sem_warp_explore:
                sem_warp_loss = tf.stack(axis=0, values=sem_warp_losses)
                sem_warp_loss = tf.reduce_mean(sem_warp_loss, 0)
                tf.summary.scalar('sem_warp_loss', model.sem_warp_loss)
                tf.summary.scalar('unit_sem_warp_loss', model.sem_warp_loss/ (opt.sem_warp_weight+tf.convert_to_tensor(1e-8, dtype=tf.float32)
) )
            if opt.sem_mask_explore:
                sem_mask_loss = tf.stack(axis=0, values=sem_mask_losses)
                sem_mask_loss = tf.reduce_mean(sem_mask_loss, 0)
                tf.summary.scalar('sem_mask_loss', model.sem_mask_loss)
                tf.summary.scalar('unit_sem_mask_loss', model.sem_mask_loss/ (opt.sem_mask_weight+tf.convert_to_tensor(1e-8, dtype=tf.float32)
) )
            if opt.sem_edge_explore:
                sem_edge_loss = tf.stack(axis=0, values=sem_edge_losses)
                sem_edge_loss = tf.reduce_mean(sem_edge_loss, 0)
                tf.summary.scalar('sem_edge_loss', model.sem_edge_loss)
                tf.summary.scalar('unit_sem_edge_loss', model.sem_edge_loss/ (opt.sem_edge_weight+tf.convert_to_tensor(1e-8, dtype=tf.float32)
) )
        
        if opt.sem_assist and opt.add_segnet:
            sem_seg_loss = tf.stack(axis=0, values=sem_seg_losses)
            sem_seg_loss = tf.reduce_mean(sem_seg_loss, 0)
            tf.summary.scalar('sem_seg_loss', sem_seg_loss)
            tf.summary.scalar('unit_sem_seg_loss', model.sem_seg_loss/ (opt.sem_seg_weight+tf.convert_to_tensor(1e-8, dtype=tf.float32)))

            ins0_seg_loss = tf.stack(axis=0, values=ins0_seg_losses)
            ins0_seg_loss = tf.reduce_mean(ins0_seg_loss, 0)
            tf.summary.scalar('ins0_seg_loss', ins0_seg_loss)
            tf.summary.scalar('unit_ins0_seg_loss', model.ins0_seg_loss/ (opt.ins0_seg_weight+tf.convert_to_tensor(1e-8, dtype=tf.float32)))

            ins1_edge_seg_loss = tf.stack(axis=0, values=ins1_edge_seg_losses)
            ins1_edge_seg_loss = tf.reduce_mean(ins1_edge_seg_loss, 0)
            tf.summary.scalar('ins1_edge_seg_loss', ins1_edge_seg_loss)
            tf.summary.scalar('unit_ins1_edge_seg_loss', model.ins1_edge_seg_loss/ (opt.ins1_edge_seg_weight+tf.convert_to_tensor(1e-8, dtype=tf.float32)))

        if opt.ins_as_loss:
            ins_loss = tf.stack(axis=0, values=ins_losses)
            ins_loss = tf.reduce_mean(ins_loss, 0)
            tf.summary.scalar('ins_loss', ins_loss)

        train_op = slim.learning.create_train_op(loss, optim,
                                                 variables_to_train=train_vars,
                                                 colocate_gradients_with_ops=True)

        # Saver
        saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                [global_step],
                                max_to_keep=opt.max_to_keep)

        merged_summary = tf.summary.merge_all()
        
        # Session
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                save_summaries_secs=0,
                                saver=None)
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            train_writer = tf.summary.FileWriter(opt.summary_dir, sess.graph)

            if opt.init_ckpt_file != None:
                sess.run(init_assign_op, init_feed_dict)
            start_time = time.time()

            for step in range(1, opt.max_steps):
                fetches = {
                    "train": train_op,
                    "global_step": global_step,
                    "incr_global_step": incr_global_step
                }
                
                if step % opt.print_interval == 0:
                    fetches["loss"] = loss
                    fetches["img_loss"]=img_loss

                    if opt.sem_as_loss:
                        fetches["sem_loss"]=sem_loss
                    if opt.ins_as_loss:
                        fetches["ins_loss"]=ins_loss
                    if opt.add_segnet:
                        fetches["sem_seg_loss"] = sem_seg_loss
                        fetches["ins0_seg_loss"] = ins0_seg_loss
                        fetches["ins1_edge_seg_loss"] = ins1_edge_seg_loss

                results = sess.run(fetches)

                #TODO Write TF Summary to file.
                if step % opt.save_summ_freq == 0:
                    step_summary = sess.run(merged_summary)
                    train_writer.add_summary(step_summary, step)

                if step % opt.print_interval == 0:

                    time_per_iter = (time.time() - start_time) / opt.print_interval
                    start_time = time.time()

                    if opt.sem_as_loss:
                        print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f ImgLoss: %.3f SemLoss: %.3f' \
                        % (step, time_per_iter, results["loss"], results["img_loss"], results["sem_loss"]))
                    elif opt.ins_as_loss:
                        print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f ImgLoss: %.3f InsLoss: %.3f' \
                        % (step, time_per_iter, results["loss"], results["img_loss"], results["ins_loss"]))
                    else:
                        print('Iteration: [%7d] | Time: %4.4fs/iter | ImgLoss: %.3f' \
                        % (step, time_per_iter, results["loss"]))

                if step % opt.save_ckpt_freq == 0:
                    saver.save(sess, os.path.join(opt.checkpoint_dir, 'model'), global_step=step)

def main(_):

    opt.num_source = opt.seq_length - 1
    opt.num_scales = 4

    opt.add_flownet = opt.mode in ['train_flow', 'test_flow']
    opt.add_dispnet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'test_depth']
    opt.add_posenet = opt.add_flownet and opt.flownet_type == 'residual' \
                      or opt.mode in ['train_rigid', 'test_pose']

    if opt.mode in ['train_rigid', 'train_flow']:
        train()
    elif opt.mode == 'test_depth':
        test_depth(opt)
    elif opt.mode == 'test_pose':
        test_pose(opt)
    elif opt.mode == 'test_flow':
        test_flow(opt)

if __name__ == '__main__':
    tf.app.run()