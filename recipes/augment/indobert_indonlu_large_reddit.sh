python scripts/run_classification.py \
    --model_name_or_path indobenchmark/indobert-large-p1 \
    --dataset_name w11wo/reddit_indonesia_sarcastic \
    --dataset_config_name default \
    --shuffle_train_dataset \
    --metric_name f1 \
    --text_column_name text \
    --label_column_name label \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0.03 \
    --label_smoothing_factor 0.0 \
    --num_train_epochs 100 \
    --do_train --do_eval --do_predict \
    --output_dir outputs/indobert-large-p1-reddit-indonesia-sarcastic-augment \
    --overwrite_output_dir \
    --hub_model_id w11wo/indobert-large-p1-reddit-indonesia-sarcastic-augment \
    --push_to_hub --hub_private_repo \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --seed 42 \
    --report_to none \
    --fp16 \
    --do_augment