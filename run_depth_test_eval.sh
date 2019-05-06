# This shell script is for SIGNet depth evaluation and visualization

## COMMAND-LINE PARAMETERS 
source $1
#MODEL_INDEX=$2

## DIR LOGIC
MODE=test_depth
DATASET_DIR=../../data/kitti_raw/
#INIT_CKPT_FILE=${CHECKPOINT_DIR}/model-${MODEL_INDEX}
INIT_CKPT_FILE=${CHECKPOINT_DIR}/model
BATCH_SIZE=1
DEPTH_TEST_SPLIT=eigen
OUTPUT_DIR=${INIT_CKPT_FILE}/output
KITTI_DIR=${DATASET_DIR}
#PRED_FILE=${OUTPUT_DIR}/model-${MODEL_INDEX}.npy
PRED_FILE=${OUTPUT_DIR}/model.npy
SPLIT=${DEPTH_TEST_SPLIT}
SEM_NUM_CLASS=19

## CUSTOMIZED PARAMETERS
SHOW_VIS=false               # Whether to visualize result

VIS_DIR=./outputs/depth/$3/  # Path to save the visualize result

LIMIT=-1                      # How many samples to save (-1 for unlimit)
INTERP=false                  # Interpolate for gt depth (To get paper's metrics, let INTERP=false)
MASK_EVAL=false
MASK_KITTI_DIR=../../data/test_files_eigen_semantic/ # Semantic segmentation for kitti test set
MASK_SUFFIX=".npy"

mkdir -p ${OUTPUT_DIR}

# RUNNING
# 1. Run the inference (test)
if [ "$#" -le 3 ];then
python3 sig_main.py \
    --mode=${MODE} \
    --dataset_dir=${DATASET_DIR} \
    --init_ckpt_file=${INIT_CKPT_FILE} \
    --batch_size=${BATCH_SIZE} \
    --scale_normalize=${SCALE_NORMALIZE} \
    --sem_assist=${SEM_ASSIST} \
    --sem_as_feat=${SEM_AS_FEAT} \
    --one_hot_sem_feat=${ONE_HOT_SEM_FEAT} \
    --sem_test_kitti_dir=${MASK_KITTI_DIR} \
    --sem_num_class=${SEM_NUM_CLASS} \
    --depth_test_split=${DEPTH_TEST_SPLIT} \
    --output_dir=${OUTPUT_DIR} \
    \
    --sem_mask_explore=${SEM_MASK_EXPLORE} \
    --sem_mask_feature=${SEM_MASK_FEATURE} \
    --sem_edge_explore=${SEM_EDGE_EXPLORE} \
    --sem_edge_feature=${SEM_EDGE_FEATURE} \
    \
    --sem_mask_pattern=${SEM_MASK_PATTERN} \
    \
    --ins_assist=${INS_ASSIST} \
    --ins_as_feat=${INS_AS_FEAT} \
    --ins_as_loss=${INS_AS_LOSS} \
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
    --block_dispnet_sem=${BLOCK_DISPNET_SEM} \
    --block_posenet_sem=${BLOCK_POSENET_SEM} \
    --new_sem_dispnet=${NEW_SEM_DISPNET} \
    --new_sem_posenet=${NEW_SEM_POSENET} \
    \
    | tee ${OUTPUT_DIR}/depth_test_log_model.txt
    #| tee ${OUTPUT_DIR}/depth_test_log_model${MODEL_INDEX}.txt
fi 

## 2. Run the evaluation
python3 kitti_eval/eval_depth_vis.py \
    --split=${SPLIT} \
    --kitti_dir=${KITTI_DIR} \
    --pred_file=${PRED_FILE} \
    --vis_dir=${VIS_DIR} \
    --interp=${INTERP} \
    --vis_limit=${LIMIT} \
    --show_vis=${SHOW_VIS} \
    --mask_eval=${MASK_EVAL} \
    --mask_kitti_dir=${MASK_KITTI_DIR} \
    --mask_suffix=${MASK_SUFFIX} \
    ${MASK_BY_CHANNEL_START_INDEX} \
    ${MASK_BY_CHANNEL_END_INDEX} \
    | tee ${OUTPUT_DIR}/depth_eval_log_model.txt
    #| tee ${OUTPUT_DIR}/depth_eval_log_model${MODEL_INDEX}.txt

