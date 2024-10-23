
# Requirements

```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

# train

```
train_cmd = (
    f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python -u src/train_bash.py --stage sft  --do_train  --model_name_or_path "
    f"{path}/LM/{llama_model} --dataset {dataset}  --dataset_dir data  --template alpaca  "
    "--finetuning_type lora  --lora_target q_proj,v_proj  --output_dir "
    f"{output_dir}  --overwrite_cache  --overwrite_output_dir  --cutoff_len "
    f"{cutoff_len}  --preprocessing_num_workers {preprocessing_num_workers}  --per_device_train_batch_size {per_device_train_batch_size}  --per_device_eval_batch_size {per_device_eval_batch_size}  "
    "--gradient_accumulation_steps 1  --lr_scheduler_type cosine  --logging_steps 10  --warmup_steps 20  "
    f"--save_steps {save_steps}  --eval_steps {eval_steps}  --evaluation_strategy steps  --load_best_model_at_end  "
    f"--learning_rate {learning_rate} --lora_rank {lora_rank}  --lora_alpha {lora_alpha}  --lora_dropout {lora_dropout} --num_train_epochs {epochs}  "
    f"--max_samples {max_samples}  --val_size {val_size} --plot_loss")
```
# test

```
test_cmd = (
    f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python -u src/train_bash.py --stage sft --do_predict --model_name_or_path "
    f"{path}/LM/{llama_model} "
    f"--adapter_name_or_path {output_dir}"
    f" --dataset {test_dataset} "
    " --dataset_dir data --template alpaca --output_dir "
    f"{mix_test_output_path} --do_sample true "
    "--temperature {temperature} --overwrite_cache --overwrite_output_dir --cutoff_len {cutoff_len} "
    "--per_device_eval_batch_size {per_device_eval_batch_size} --max_new_tokens {max_new_tokens} --predict_with_generate")
```

# merge model

```
#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES=0 python ../../src/export_model.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B/lora/sft \
    --template default \
    --finetuning_type lora \
    --export_dir ../../models/llama2-7b-sft \
    --export_size 2 \
    --export_legacy_format False
```


# link
- https://github.com/hiyouga/LLaMA-Factory