export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="3"
NUM_PROCESSES=1
MASTER_PORT=29508

OUTPUT_DIR="999_SGH_adapter_SG+high"
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

DATA_DIR=data/shadow_gen
TEST_FILE=test_sort.jsonl

accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/inference/main_SGH_adapter.py \
    --pretrained_model_name_or_path checkpoints/base \
    --pretrained_vae_model_name_or_path checkpoints/condition_vae \
	--pretrained_unet_model_name_or_path 999_train_Harmony_final_SG+lossSG/weights-34800 \
    --pretrained_unet_Inpainting_model_name_or_path checkpoints/base/unet_Inpainting \
    --pretrained_adapter_model_name_or_path 999_train_adapter_highPer+highCos/weights-15000 \
	--stage2_model_name_or_path checkpoints/refinement \
    --dataset_root $DATA_DIR \
	--test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
	--seed=42 \
	--resolution=512 \
	--output_resolution=512 \
	--eval_batch_size=10 \
	--dataloader_num_workers=8 \
	--mixed_precision="fp16" \
    --adapter_type "Adapter"
	

	# --stage2_model_name_or_path "" \
	# --rank=4 \
	# --stage2_model_name_or_path checkpoints/refinement \