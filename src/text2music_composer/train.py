# # Without using
# !python dreambooth_musicgen.py \
#     --model_name_or_path "./ckpts/facebook-musicgen-small" \
#     --dataset_name "CLAPv2-MusicCaps" \
#     --dataset_config_name "default" \
#     --target_audio_column_name "audio" \
#     --text_column_name "caption" \
#     --train_split_name "train" \
#     --eval_split_name "train" \
#     --output_dir "./ckpts/MusicGen-Small-MusicCaps-finetuning" \
#     --do_train \
#     --fp16 \
#     --num_train_epochs 700 \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --per_device_train_batch_size 16 \
#     --learning_rate 1e-5 \
#     --adam_beta1 0.9 \
#     --adam_beta2 0.99 \
#     --weight_decay 0.1 \
#     --guidance_scale 3.0 \
#     --do_eval \
#     --predict_with_generate \
#     --include_inputs_for_metrics \
#     --eval_steps 25 \
#     --per_device_eval_batch_size 8 \
#     --max_eval_samples 64 \
#     --generation_max_length 400 \
#     --dataloader_num_workers 8 \
#     --logging_steps 1 \
#     --max_duration_in_seconds 30 \
#     --min_duration_in_seconds 1.0 \
#     --preprocessing_num_workers 4 \
#     --pad_token_id 2048 \
#     --decoder_start_token_id 2048 \
#     --seed 456 \
#     --overwrite_output_dir \
#     --push_to_hub false \
#     --save_total_limit 1 \
#     --report_to none



device = "cuda" if torch.is available else "cpu"
!python src/audioldm/train.py \
    --config_yaml configs/my_configs/audioldm_original.yaml \
    --reload_from_ckpt ckpts/AudioLDM/audioldm-s-full.ckpt \
    --wandb_off --accelerator "$device"