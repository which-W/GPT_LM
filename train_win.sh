python3 train.py \
  --train_data_path .\data\TinyStories-train.bin \
  --valid_data_path .\data\TinyStories-valid.bin \
  --batch_size 32 \
  --use_wandb \
  --wandb_project "tinystories-transformer" \
  --wandb_run_name "fourth-experiment"