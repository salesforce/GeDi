

export lambda_=0.80
export lr=2e-5

#final GeDi LM checkpoint saved at --output_dir
python ../train_GeDi.py --task_name SST-2 \
  --overwrite_output_dir \
  --do_eval  \
  --do_train \
  --logit_scale \
  --data_dir ../data/AG-news  \
  --max_seq_length 192 \
  --overwrite_cache \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size  8 \
  --learning_rate $lr  \
  --num_train_epochs 1.0  \
  --output_dir ../topic_GeDi_retrained \
  --model_type gpt2  \
  --model_name_or_path gpt2-medium \
  --gen_weight $lambda_ \
  --logging_steps 500 \
  --save_steps 5000000000 \
  --code_0 false \
  --code_1 true \
  --fp16

