export NCCL_P2P_LEVEL=NVL
export PYTHONPATH=.:$PYTHONPATH

ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_PROCESSES=4
MASTER_PORT=29500

OUTPUT_DIR="999_train_controlnet_SGmask"
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/train/inpainting_controlnet.py \
	--pretrained_model_name_or_path "checkpoints/base" \
	--vae_path "checkpoints/condition_vae" \
	--output_dir $OUTPUT_DIR \
	--max_train_steps=20000 \
	--seed=42 \
	--train_batch_size=8 \
	--eval_batch_size=8 \
	--dataloader_num_workers=8 \
	--gradient_accumulation_steps=2 \
	--learning_rate=1e-5 \
	--lr_scheduler "constant" \
	--lr_warmup_ratio=0 \
	--adam_weight_decay=1e-2 \
	--mixed_precision="fp16" \
	--checkpointing_epochs=5 \
	--checkpoints_total_limit=5 \
	--dataset_root "data/shadow_gen" \
	--train_file "train.jsonl" \
	--test_file "test.jsonl" \
	--resolution=512 \
	--infer_resolution=512 \
	--image_log_interval=200 \
	--gradient_checkpointing \
	--enable_xformers_memory_efficient_attention \
    --use_ema \
	--lambda_loss_local=1.0 \
	--lambda_loss_per=0.01 \
	--lambda_loss_cos=0.01

	# --max_train_steps=150000 \
    # --num_train_epochs=50 \
    # --use_ema \
	# --nofg \
	# --lambda_loss_whole=0.0 \