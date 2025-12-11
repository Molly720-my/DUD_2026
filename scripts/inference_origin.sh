export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
NUM_PROCESSES=6
MASTER_PORT=29505

OUTPUT_DIR="output_origin"
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

DATA_DIR=data/iHarmony4
TEST_FILE=test.jsonl

accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/inference/main_origin.py \
    --pretrained_model_name_or_path checkpoints/base \
    --pretrained_vae_model_name_or_path checkpoints/condition_vae \
    --pretrained_unet_model_name_or_path checkpoints/base/unet_Inpainting \
	--stage2_model_name_or_path checkpoints/refinement \
    --dataset_root $DATA_DIR \
	--test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
	--seed=42 \
	--resolution=512 \
	--output_resolution=512 \
	--eval_batch_size=5 \
	--dataloader_num_workers=10 \
	--mixed_precision="fp16" \
	--inverse

	# --stage2_model_name_or_path checkpoints/refinement \ ""