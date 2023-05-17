python model_train.py --eft_mode lora:8:64 --run_name lora_finetuning_r8_a64 --model_name EleutherAI/pythia-70m --input_train_path ./train.moonscript.seq512.json --input_dev_path ./dev.moonscript.seq512.json --output_dir ./logs/eft --do_train --do_eval --evaluation_strategy epoch --per_device_train_batch_size 128 --warmup_ratio 0.05 --save_strategy epoch  --load_best_model_at_end --metric_for_best_model loss --report_to wandb --num_train_epochs 64 --save_total_limit 10