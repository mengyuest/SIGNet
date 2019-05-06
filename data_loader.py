# Mostly based on the code written by Tinghui Zhou & Clement Godard: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data_loader.py
# https://github.com/mrharicot/monodepth/blob/master/monodepth_dataloader.py
from __future__ import division
import os
import random
import tensorflow as tf

import numpy as np
def read_npy_file(item):
    data = np.load(item.decode())
    return data.astype(np.uint8)

class DataLoader(object):
    def __init__(self, opt=None,):
        self.opt = opt
        seed = 8964
        tf.set_random_seed(seed)
        random.seed(seed)

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        opt = self.opt

        # Load the list of training files into queues
        #TODO 
        if opt.train_lite:
            file_list = self.format_file_list(opt.dataset_dir, opt.filelist_dir, 'train_lite')
        else:
            file_list = self.format_file_list(opt.dataset_dir, opt.filelist_dir, 'train')
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'], shuffle=False)
        cam_paths_queue = tf.train.string_input_producer(
            file_list['cam_file_list'], shuffle=False)
        
        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents, 
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, opt.img_height, opt.img_width, opt.num_source)

        #TODO Load Semantics
        #     See cityscape label defs in https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L62
        #     Also notice that deeplabv3+ uses `train_id` https://github.com/tensorflow/models/blob/69b016449ffc797421bf003d8b7fd8545db866d7/research/deeplab/datasets/build_cityscapes_data.py#L46
        #     Color maps are in https://github.com/tensorflow/models/blob/69b016449ffc797421bf003d8b7fd8545db866d7/research/deeplab/utils/get_dataset_colormap.py#L207
        if opt.sem_assist:
            sem_paths_queue = tf.train.string_input_producer(
                file_list['sem_image_file_list'], shuffle=False)
            sem_reader = tf.WholeFileReader()
            sem_keys, sem_contents = sem_reader.read(sem_paths_queue)

            if opt.load_from_raw:
                sem_seq = tf.reshape(tf.decode_raw(sem_contents, tf.uint8), [1, opt.img_height, (opt.num_source+1) * opt.img_width])
            else:
                sem_seq = tf.py_func(read_npy_file, [sem_keys], [tf.uint8,])
        
        #TODO Load Instances: we use COCO
        #     Two channels: class and id level. For id level we only use the edge
        if opt.ins_assist:
            ins_paths_queue = tf.train.string_input_producer(
                file_list['ins_image_file_list'], shuffle=False)
            ins_reader = tf.WholeFileReader()
            ins_keys, ins_contents = ins_reader.read(ins_paths_queue)

            if opt.load_from_raw:
                ins_seq = tf.reshape(tf.decode_raw(ins_contents, tf.uint8), [1, opt.img_height, (opt.num_source+1) * opt.img_width, 2])
            else:
                ins_seq = tf.py_func(read_npy_file, [ins_keys], [tf.uint8,])


        #TODO 1. SHUFFLE BATCH
        # Form training batches
        seed = random.randint(0, 2**31 - 1)
        min_after_dequeue = 2048
        capacity = min_after_dequeue + opt.num_threads * opt.batch_size

        if opt.sem_assist and opt.ins_assist:
            src_image_stack, tgt_image, intrinsics, sem_seq, ins_seq = tf.train.shuffle_batch(
                [src_image_stack, tgt_image, intrinsics, sem_seq[0], ins_seq[0]],
                        opt.batch_size, capacity, min_after_dequeue, opt.num_threads, seed)
        
        elif opt.sem_assist:
            src_image_stack, tgt_image, intrinsics, sem_seq = tf.train.shuffle_batch(
                [src_image_stack, tgt_image, intrinsics, sem_seq[0]],
                        opt.batch_size, capacity, min_after_dequeue, opt.num_threads, seed)
        
        elif opt.ins_assist:
            src_image_stack, tgt_image, intrinsics, ins_seq = tf.train.shuffle_batch(
                [src_image_stack, tgt_image, intrinsics, ins_seq[0]],
                        opt.batch_size, capacity, min_after_dequeue, opt.num_threads, seed)
        
        else:
            src_image_stack, tgt_image, intrinsics = tf.train.shuffle_batch(
                [src_image_stack, tgt_image, intrinsics],
                        opt.batch_size, capacity, min_after_dequeue, opt.num_threads, seed)

        # semantic segmentation
        tgt_sem=None
        tgt_sem_map=None
        tgt_sem_mask=None
        tgt_sem_edge=None
        src_sem_stack=None
        src_sem_map_stack=None
        src_sem_mask_stack=None
        src_sem_edge_stack=None

        # ins0 ~ instance level, but still class segmentation
        tgt_ins0=None
        tgt_ins0_map=None
        tgt_ins0_edge=None
        src_ins0_stack=None
        src_ins0_map_stack=None
        src_ins0_edge_stack=None

        # ins1 ~ instance level, but this is id segmentation
        tgt_ins1_edge=None
        src_ins1_edge_stack=None

        #TODO 2. TRAMSFORMATION AND UNPACKING
        if opt.sem_assist:
            #TODO get one-hot encoded         sem_oh_seq (4,128,1248,19)X{0,1}
            sem_oh_seq = tf.cast(tf.one_hot(sem_seq, on_value=1, depth = opt.sem_num_class), tf.uint8)
            #TODO decouple   tgt_sem (4,128,1248,19)X{0,1}   src_sem_stack (4,128,1248,2*19)X{0,1}
            tgt_sem, src_sem_stack = self.unpack_sem_sequence_batch_atom(sem_oh_seq, opt.sem_num_class)

            #TODO get densemap     sem_map_seq (4,128,1248,1)X{0,1,...,18}
            sem_map_seq = tf.expand_dims(sem_seq,-1)
            #TODO decouple   tgt_sem_map (4,128,1248,1)X{0,1,...,18}   src_sem_map_stack (4,128,1248,2*1)X{0,1,...,18}
            tgt_sem_map, src_sem_map_stack = self.unpack_sem_sequence_batch_atom(sem_map_seq, 1)

            if opt.sem_mask_explore: 
                #TODO get sem mask   sem_mask_seq (4,128,1248,c) here we assume c=1 
                sem_mask_seq = self.get_sem_mask_batch(sem_seq)
                #TODO decouple   tgt_sem_mask (4,128,1248,c)   src_sem_mask_stack (4,128,1248,2*c)
                tgt_sem_mask, src_sem_mask_stack = self.unpack_sem_sequence_batch_atom(sem_mask_seq, 1)
            
            if opt.sem_edge_explore:
                #TODO get sem edge   sem_edge_seq (4,128,1248,c) here we assume c=1 
                sem_edge_seq=self.get_sem_edge_batch(sem_seq)
                #TODO decouple   tgt_sem_edge (4,128,1248,c)   src_sem_edge_stack (4,128,1248,2*c)
                tgt_sem_edge, src_sem_edge_stack = self.unpack_sem_sequence_batch_atom(sem_edge_seq, 1)

        if opt.ins_assist:
            ins0_seq = ins_seq[:,:,:,0]
            ins1_seq = ins_seq[:,:,:,1]

            #TODO get one-hot  ins0_oh_seq (4,128,1248,81)X{0,1}
            ins0_oh_seq = tf.cast(tf.one_hot(ins0_seq, on_value=1, depth = opt.ins_num_class), tf.uint8)
            #ins1_oh_seq = tf.cast(tf.one_hot(ins1_seq, on_value=1, depth = 255), tf.uint8)
            
            #TODO decouple   tgt_ins0 (4,128,1248,81)X{0,1}   src_ins0_stack (4,128,1248,2*81)X{0,1}
            tgt_ins0, src_ins0_stack = self.unpack_sem_sequence_batch_atom(ins0_oh_seq, opt.ins_num_class)
            #tgt_ins1, src_ins1_stack = self.unpack_sem_sequence_batch_atom(ins1_oh_seq, opt.ins_num_class)

            #TODO get densemap  sem_ins0_seq (4,128,1248,1)X{0,1,...,80}
            ins0_map_seq = ins_seq[:,:,:,:1]
            ins1_map_seq = ins_seq[:,:,:,1:]

            #TODO decouple  tgt_ins0_map (4,128,1248,1)X{0,1,...,80}  src_ins0_map_stack (4,128,1248,2*1)X{0,1,...,80}
            tgt_ins0_map, src_ins0_map_stack = self.unpack_sem_sequence_batch_atom(ins0_map_seq, 1)
            tgt_ins1_map, src_ins1_map_stack = self.unpack_sem_sequence_batch_atom(ins1_map_seq, 1)

            if opt.ins0_edge_explore:
                #TODO get edge   ins0_edge_seq (4,128,1248,c)  here we assume c=1 
                ins0_edge_seq=self.get_sem_edge_batch(ins0_seq)
                #TODO decouple   tgt_ins0_edge (4,128,1248,c)  src_ins0_edge_stack (4,128,1248,2*c)
                tgt_ins0_edge, src_ins0_edge_stack = self.unpack_sem_sequence_batch_atom(ins0_edge_seq, 1)
            
            if opt.ins1_edge_explore:
                #TODO get edge   ins1_edge_seq (4,128,1248,c) here we assume c=1 
                ins1_edge_seq=self.get_sem_edge_batch(ins1_seq)
                #TODO decouple   tgt_ins1_edge (4,128,1248,c)   src_ins1_edge_stack (4,128,1248,2*c)
                tgt_ins1_edge, src_ins1_edge_stack = self.unpack_sem_sequence_batch_atom(ins1_edge_seq, 1)

        #TODO 3. DATA AUGMENTATION
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, intrinsics, aug_params = self.data_augmentation(
            image_all, intrinsics, opt.img_height, opt.img_width) #TODO changed API
        
        if opt.sem_assist:
            ##TODO Do the same data augmentation for semantic segmentations
            tgt_sem, src_sem_stack = self.data_aug(tgt_sem, src_sem_stack, aug_params, "bilinear")
            tgt_sem_map, src_sem_map_stack = self.data_aug(tgt_sem_map, src_sem_map_stack, aug_params, "neighbor")
            
            if self.opt.sem_mask_explore:
                tgt_sem_mask, src_sem_mask_stack = \
                    self.data_aug(tgt_sem_mask, src_sem_mask_stack, aug_params, "bilinear")
            
            if self.opt.sem_edge_explore:
                tgt_sem_edge, src_sem_edge_stack = \
                    self.data_aug(tgt_sem_edge, src_sem_edge_stack, aug_params, "bilinear")
                    #TODO maybe transfer needs this settings self.data_aug(tgt_sem_edge, src_sem_edge_stack, aug_params, "neighbor")

        if opt.ins_assist:
            ##TODO Do the same data augmentation for instance segmentations
            tgt_ins0, src_ins0_stack = self.data_aug(tgt_ins0, src_ins0_stack, aug_params, "bilinear")
            #tgt_ins1, src_ins1_stack = self.data_aug(tgt_ins1, src_ins1_stack, aug_params, "bilinear")

            tgt_ins0_map, src_ins0_map_stack = self.data_aug(tgt_ins0_map, src_ins0_map_stack, aug_params, "neighbor")
            #tgt_ins1_map, src_ins1_map_stack = self.data_aug(tgt_ins1_map, src_ins1_map_stack, aug_params, "neighbor")
            
            if self.opt.ins0_edge_explore:
                tgt_ins0_edge, src_ins0_edge_stack = \
                    self.data_aug(tgt_ins0_edge, src_ins0_edge_stack, aug_params, "bilinear")
                    #TODO maybe transfer needs this settings self.data_aug(tgt_ins0_edge, src_ins0_edge_stack, aug_params, "neighbor")
                    
            
            if self.opt.ins1_edge_explore:
                tgt_ins1_edge, src_ins1_edge_stack = \
                    self.data_aug(tgt_ins1_edge, src_ins1_edge_stack, aug_params, "bilinear")
                    #TODO maybe transfer needs this settings self.data_aug(tgt_ins1_edge, src_ins1_edge_stack, aug_params, "neighbor")
                    
        # 4. RETURN
        # image_channels=3*opt.seq_length
        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:] #3:image_channels]
        intrinsics = self.get_multi_scale_intrinsics(intrinsics, opt.num_scales)

        # if opt.sem_assist and opt.ins_assist:
        return tgt_image, src_image_stack, intrinsics, \
                [tgt_sem, tgt_sem_map, tgt_sem_mask, tgt_sem_edge], \
                [src_sem_stack, src_sem_map_stack, src_sem_mask_stack, src_sem_edge_stack], \
                [tgt_ins0, tgt_ins0_map, tgt_ins0_edge, tgt_ins1_edge], \
                [src_ins0_stack, src_ins0_map_stack, src_ins0_edge_stack, src_ins1_edge_stack]

    def data_aug(self, tgt_sem, src_sem_stack, aug_params, f_str):

        if f_str=="bilinear":
            resize_f=tf.image.resize_bilinear
        elif f_str=="neighbor":
            resize_f=tf.image.resize_nearest_neighbor
        else:
            print("dont know this resizing method, exit...")
            exit()

        # 1. fetch params
        out_hw, yxhw = aug_params
        offset_y,offset_x,out_h,out_w = yxhw

        # 2. random scaling
        tgt_sem = resize_f(tgt_sem, out_hw)
        src_sem_stack = resize_f(src_sem_stack, out_hw)

        # 3. random cropping
        tgt_sem = tf.image.crop_to_bounding_box(
            tgt_sem, offset_y, offset_x, out_h, out_w)
        src_sem_stack = tf.image.crop_to_bounding_box(
            src_sem_stack, offset_y, offset_x, out_h, out_w)
        
        # 4. cast
        if self.opt.data_aug_cast:
            tgt_sem = tf.cast(tgt_sem, dtype=tf.uint8)
            src_sem_stack=tf.cast(src_sem_stack, dtype=tf.uint8)

        return tgt_sem, src_sem_stack

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    #TODO change the api parameters, add one to the return value to make sure segmentation aug the save way
    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics, [out_h,out_w]

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics, [offset_y, offset_x,out_h,out_w]

        # Random coloring
        def random_coloring(im):
            batch_size, in_h, in_w, in_c = im.get_shape().as_list()
            im_f = tf.image.convert_image_dtype(im, tf.float32)

            # randomly shift gamma
            random_gamma = tf.random_uniform([], 0.8, 1.2)
            im_aug  = im_f  ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random_uniform([], 0.5, 2.0)
            im_aug  =  im_aug * random_brightness

            # randomly shift color
            random_colors = tf.random_uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
            im_aug  *= color_image

            # saturate
            im_aug  = tf.clip_by_value(im_aug,  0, 1)

            im_aug = tf.image.convert_image_dtype(im_aug, tf.uint8)

            return im_aug
        im, intrinsics, out_hw = random_scaling(im, intrinsics)
        im, intrinsics, yxhw = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        do_augment  = tf.random_uniform([], 0, 1)
        im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)
            
        return im, intrinsics, [out_hw, yxhw]

    def format_file_list(self, data_root, filelist_root, split):
        with open(filelist_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        
        #TODO add sem file list
        if self.opt.sem_assist:
            if self.opt.load_from_raw:
                sem_file_list = [os.path.join(data_root, subfolders[i], 
                    frame_ids[i] + '.raw') for i in range(len(frames))]
            else:
                sem_file_list = [os.path.join(data_root, subfolders[i], 
                    frame_ids[i] + '.npy') for i in range(len(frames))]
        
        #TODO add ins file list
        if self.opt.ins_assist:
            if self.opt.load_from_raw:
                ins_file_list = [os.path.join(self.opt.ins_train_kitti_dir, subfolders[i], 
                    frame_ids[i] + '_instance_new.raw') for i in range(len(frames))]
            else:
                ins_file_list = [os.path.join(self.opt.ins_train_kitti_dir, subfolders[i], 
                    frame_ids[i] + '_instance_new.npy') for i in range(len(frames))]

        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list

        #TODO add sem image path
        if self.opt.sem_assist:
            all_list['sem_image_file_list']=sem_file_list 

        #TODO add ins_image_path
        if self.opt.ins_assist:
            all_list["ins_image_file_list"]=ins_file_list
        
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, tgt_start_idx, 0], 
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, int(tgt_start_idx + img_width), 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, i*img_width, 0], 
                                    [-1, img_width, -1]) 
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height, 
                                   img_width, 
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack
    

    #TODO unpack semantic
    # Shape (batch_size, sem_height, sem_width*(num_source+1), num_channel)
    # Return tgt and src things
    # They are in (bs, h, w, ch) and (bs, h, w, ch*num_source)
    def unpack_sem_sequence_batch_atom(self, sem_seq, num_channel):

        sem_height = self.opt.img_height
        sem_width = self.opt.img_width
        num_source = self.opt.num_source
        batch_size =self.opt.batch_size

        # Assuming the center sem image is the target frame
        tgt_start_idx = int(sem_width * (num_source//2))
        tgt_sem = sem_seq[:, :, tgt_start_idx: tgt_start_idx + sem_width, :]
        
        # Source frames before the target frame
        src_sem_1 = sem_seq[:, :, :tgt_start_idx, :]

        # Source frames after the target frame
        src_sem_2 = sem_seq[:, :, tgt_start_idx+sem_width:, :]
        
        # Still get a fat tensor, but only src related
        src_sem_seq = tf.concat([src_sem_1, src_sem_2], axis=2)
        
        # Then we get a thick tensor stacking source along channels (i.e. [batchsize, H, W, N*K])
        src_sem_stack = tf.concat([src_sem_seq[:, :, i*sem_width:(i+1)*sem_width, :]
                                    for i in range(num_source)], axis=3)
        
        # Set the shape
        tgt_sem.set_shape([batch_size, sem_height, sem_width, num_channel])
        src_sem_stack.set_shape([batch_size, sem_height, sem_width, num_source * num_channel])
        
        return tgt_sem, src_sem_stack

    
    def get_sem_mask_batch(self, sem_seq): #4*128*(416*3=1248) X {0,1,..,18} =>  4*128*1248*c
        if self.opt.sem_mask_pattern=="foreground":
            return tf.expand_dims(tf.cast(sem_seq>10,tf.uint8),-1)
        else:
            return tf.expand_dims(tf.cast(sem_seq<=10,tf.uint8),-1)
    
    
    def get_sem_edge_batch(self, sem_seq): #4*128*(416*3=1248) X {0,1,..,18} =>  4*128*1248*c

        def some_f(seg):
            x = tf.pad(1-tf.cast(tf.equal(seg[:, :, :-1], seg[:, :, 1:]), dtype=tf.uint8), [[0, 0], [0, 0], [0, 1]])
            y = tf.pad(1-tf.cast(tf.equal(seg[:, :-1, :], seg[:, 1:, :]), dtype=tf.uint8), [[0, 0], [0, 1], [0, 0]])
            return x + y            

        width=self.opt.img_width
        num_source=self.opt.num_source
        tgt_start = int(width * (num_source//2))
        
        tf_seg0 = sem_seq[:,:,tgt_start:tgt_start+width] 
        tf_seg1 = sem_seq[:,:,:tgt_start] 
        tf_seg2 = sem_seq[:,:,tgt_start + width:] 
        tf_edge0=some_f(tf_seg0)
        tf_edge1=some_f(tf_seg1)
        tf_edge2=some_f(tf_seg2)
        
        return tf.expand_dims(tf.concat([tf_edge0, tf_edge1, tf_edge2], axis=2),-1)


    #TODO tensorflow shuffle batch based on our condition
    def tf_shuffle_batch(self, shuffle_list, batch_size, capacity, min_after_dequeue, num_threads, seed):
        src_image_stack=tgt_image=intrinsics=src_sem_stack=tgt_sem=src_sem_mask_stack=tgt_sem_mask=src_sem_edge_stack=tgt_sem_edge = None
        return src_image_stack, tgt_image, intrinsics, src_sem_stack, tgt_sem, src_sem_mask_stack, tgt_sem_mask, src_sem_edge_stack, tgt_sem_edge


    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale