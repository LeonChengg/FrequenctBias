---
library_name: peft
license: other
base_model: /mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: llama3.2-1b-lora-EG
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama3.2-1b-lora-EG

This model is a fine-tuned version of [/mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct](https://huggingface.co//mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct) on the EG_entailment_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3045

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- total_eval_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.028         | 3.0676 | 500  | 0.2031          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.4.0+cu121
- Datasets 2.20.0
- Tokenizers 0.21.4