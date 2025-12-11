export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="2,3"
NUM_PROCESSES=2
MASTER_PORT=29501

DATASET_ROOT=data/shadow_gen/desobav2
OUTPUT_DIR=$DATASET_ROOT/cand_composite_images
TEST_FILE="inverse.jsonl"


accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/inference/inverse.py \
	--pretrained_model_name_or_path "checkpoints/inverse" \
	--pretrained_vae_model_name_or_path "checkpoints/condition_vae" \
	--pretrained_unet_model_name_or_path "output_train_inverse/weights-47300" \
	--dataset_root $DATASET_ROOT \
	--test_file $TEST_FILE \
	--output_dir $OUTPUT_DIR \
	--seed=42 \
	--resolution=512 \
	--output_resolution=512 \
	--eval_batch_size=8 \
	--dataloader_num_workers=8 \
	--mixed_precision="fp16" \
	--rounds=10