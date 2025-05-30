swift sft \
  --model /mnt/ali-sh-1/dataset/zeus/cache/modelscope/hub/AIDC-AI/Ovis2-8B \
  --dataset news_gen_cls_20250329_train \
  --val_dataset news_gen_cls_20250329_test \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --torch_dtype bfloat16 \
  --ddp_backend nccl \
  --dataset_num_proc 32 \
  --dataloader_num_workers 16 \
  --load_from_cache_file true \
  --output_dir ./news_gen_ovis_cls_sft