# This shell script is for depth training

if [ "$#" -ne 1 ]; then
    echo "please give the configuration file path"
    echo 'Usage: sh run_depth_train.sh foobar/toy.cfg'
    exit 1
fi

if [ "$#" == 1 ]; then
    echo "source from: $1"
    source $1
fi

mkdir -p ${CHECKPOINT_DIR}/logs
cp -r config ${CHECKPOINT_DIR}/logs/

## Clarify the parameters
PY_CMD="python3 sig_main.py \
    --mode=train_rigid \
    --dataset_dir=${DATASET_DIR} \
    ${SET_INIT_CKPT} \
    --batch_size=${BATCH_SIZE} \
    --num_threads=${NUM_THREADS} \
    --seq_length=${SEQ_LENGTH} \
    \
    --num_gpus=${NUM_GPUS} \
    --filelist_dir=${FILELIST_DIR} \
    --enable_batch_norm=${ENABLE_BATCH_NORM}\
    --batch_norm_is_training=${BATCH_NORM_IS_TRAINING}\
    \
    --train_lite=${TRAIN_LITE} \
    --print_interval=${PRINT_INTERVAL} \
    \
    --summary_dir=${SUMMARY_DIR} \
    --save_summ_freq=${SAVE_SUMM_FREQ} \
    --max_outputs=${MAX_OUTPUTS} \
    \
    --use_regularization=${USE_REGULARIZATION} \
    \
    --sem_assist=${SEM_ASSIST} \
    --load_from_raw=${LOAD_FROM_RAW} \
    --sem_num_class=${SEM_NUM_CLASS} \
    --sem_as_loss=${SEM_AS_LOSS} \
    --sem_as_feat=${SEM_AS_FEAT} \
    --one_hot_sem_feat=${ONE_HOT_SEM_FEAT} \
    --fixed_posenet=${FIXED_POSENET} \
    \
    --data_aug_cast=${DATA_AUG_CAST} \
    \
    --sem_warp_explore=${SEM_WARP_EXPLORE} \
    --sem_warp_function=${SEM_WARP_FUNCTION} \
    --sem_warp_weight=${SEM_WARP_WEIGHT} \
    --sem_edge_explore=${SEM_EDGE_EXPLORE} \
    --sem_edge_feature=${SEM_EDGE_FEATURE} \
    --sem_edge_pattern=${SEM_EDGE_PATTERN} \
    --sem_edge_function=${SEM_EDGE_FUNCTION} \
    --sem_edge_weight=${SEM_EDGE_WEIGHT} \
    --sem_mask_explore=${SEM_MASK_EXPLORE} \
    --sem_mask_pattern=${SEM_MASK_PATTERN} \
    --sem_mask_function=${SEM_MASK_FUNCTION} \
    --sem_mask_weight=${SEM_MASK_WEIGHT} \
    --sem_mask_feature=${SEM_MASK_FEATURE} \
    \
    --add_segnet=${ADD_SEGNET} \
    --transfer_network_structure=${TRANSFER_NETWORK_STRUCTURE} \
    --sem_seg_weight=${SEM_SEG_WEIGHT} \
    --ins0_seg_weight=${INS0_SEG_WEIGHT} \
    --ins1_edge_seg_weight=${INS1_EDGE_SEG_WEIGHT} \
    --sem_mask_pattern=${SEM_MASK_PATTERN} \
    --transfer_learn_sem=${TRANSFER_LEARN_SEM} \
    --transfer_learn_ins0=${TRANSFER_LEARN_INS0} \
    --transfer_learn_ins1_edge=${TRANSFER_LEARN_INS1_EDGE} \
    \
    --block_dispnet_sem=${BLOCK_DISPNET_SEM} \
    --block_posenet_sem=${BLOCK_POSENET_SEM} \
    --new_sem_dispnet=${NEW_SEM_DISPNET} \
    --new_sem_posenet=${NEW_SEM_POSENET} \
    \
    --ins_assist=${INS_ASSIST} \
    --ins_num_class=${INS_NUM_CLASS}\
    --ins_as_loss=${INS_AS_LOSS} \
    --ins_l2_norm_weight=${INS_L2_NORM_WEIGHT} \
    \
    --ins_as_feat=${INS_AS_FEAT} \
    --ins0_dense_feature=${INS0_DENSE_FEATURE} \
    --ins0_onehot_feature=${INS0_ONEHOT_FEATURE} \
    --ins0_edge_explore=${INS0_EDGE_EXPLORE} \
    --ins0_edge_feature=${INS0_EDGE_FEATURE} \
    --ins1_dense_feature=${INS1_DENSE_FEATURE} \
    --ins1_onehot_feature=${INS1_ONEHOT_FEATURE} \
    --ins1_edge_explore=${INS1_EDGE_EXPLORE} \
    --ins1_edge_feature=${INS1_EDGE_FEATURE} \
    --ins_train_kitti_dir=${INS_TRAIN_KITTI_DIR} \
    --ins_test_kitti_dir=${INS_TEST_KITTI_DIR} \
    \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --learning_rate=${LEARNING_RATE} \
    --max_steps=${MAX_STEPS} \
    --save_ckpt_freq=${SAVE_CKPT_FREQ} \
    \
    --scale_normalize=${SCALE_NORMALIZE} \
    --rigid_warp_weight=${RIGID_WARP_WEIGHT} \
    --disp_smooth_weight=${DISP_SMOOTH_WEIGHT} \
    | tee ${CHECKPOINT_DIR}/logs/depth_train_log.txt"


## Save parameters details into this log file
echo ${PY_CMD} > ${CHECKPOINT_DIR}/logs/depth_train_cmd.txt

## Start the training
eval ${PY_CMD}


