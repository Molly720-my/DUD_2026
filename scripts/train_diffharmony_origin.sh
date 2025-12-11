export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=.:$PYTHONPATH

ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="2,3"
NUM_PROCESSES=2
MASTER_PORT=29507

OUTPUT_DIR="output_train_inverse_2"
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/train/diffharmony_origin.py \
	--pretrained_model_name_or_path "checkpoints/base" \
	--vae_path "checkpoints/condition_vae" \
	--output_dir $OUTPUT_DIR \
	--max_train_steps=150000 \
	--seed=42 \
	--train_batch_size=20 \
	--eval_batch_size=20 \
	--dataloader_num_workers=8 \
	--gradient_accumulation_steps=3 \
	--learning_rate=1e-5 \
	--lr_scheduler "constant" \
	--lr_warmup_ratio=0 \
	--use_ema \
	--adam_weight_decay=1e-2 \
	--mixed_precision="fp16" \
	--checkpointing_epochs=5 \
	--checkpoints_total_limit=5 \
	--dataset_root "data/iHarmony4" \
	--train_file "train.jsonl" \
	--test_file "test.jsonl" \
	--resolution=512 \
	--infer_resolution=512 \
	--image_log_interval=100 \
	--gradient_checkpointing \
	--enable_xformers_memory_efficient_attention \
	--mode "inverse" \
	--resume_from_checkpoint "latest"

	# --max_train_steps=150000 \
	# --num_train_epochs=100 \
	# --resume_from_checkpoint "latest"