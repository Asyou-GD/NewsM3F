from dataclasses import dataclass, field
import transformers
import trl
from typing import List, Optional
import os
import mmengine


qs_cloud = 'search01'  # 'search01' or 'nlp-ali' or 'nj-larc'
time_data = '2024-08-12'
task_name = '夸大宣传'  # '标题党' or '表意不规范' or '夸大宣传'
# task_name = '夸大宣传'  # '标题党' or '表意不规范' or '夸大宣传'
dataset_name = 'trainval_for_text_model'
# work_dir = f'work_dirs/quality/Qwen2-1.5B_title_{time_data}'
work_dir = f'work_dirs/quality/Qwen2-1.5B_xuanchuan_{time_data}'
os.environ['WANDB_PROJECT'] = f'XHS'

model_path = f'work_dirs/model_cache/Qwen2-1.5B-Instruct'
dataset_path = f'/mnt/{qs_cloud}/dataset/cky_data/notemarket/note_order_trainvaltest_data_df_{time_data.replace("-", "")}'
dataset_path = dataset_path + f'/{dataset_name}'

os.environ["WANDB_API_KEY"] = "bd1b24e183d6fe3c14ecdbcc8d07c7dd0ec540e5"
os.environ['WANDB_DIR'] = f'{work_dir}/wandb'
mmengine.mkdir_or_exist(os.environ['WANDB_DIR'])


@dataclass
class SFTScriptArguments(trl.SFTScriptArguments):
	task_name: str = field(default=task_name)
	dataset_name: str = field(default=dataset_path, metadata={"help": "Path to the training data."})
	dataset_train_split: str = field(default="train", metadata={"help": "The name of the training split."})
	dataset_test_split: str = field(default="test", metadata={"help": "The name of the test split."})
	dataset_resample: bool = field(default=True)


@dataclass
class SFTConfig(trl.SFTConfig, transformers.TrainingArguments):
	output_dir: str = field(default=work_dir)
	run_name: str = field(default=work_dir.split('/')[-1])
	log_level: str = field(default="debug")
	load_best_model_at_end: bool = field(default=True)
	metric_for_best_model: str = field(default="Type1_R_at_P60")
	greater_is_better: bool = field(default=True)
	report_to: str = field(default="none")
	optim: str = field(default="adamw_torch")
	dataset_text_field: str = field(default=None)
	learning_rate: float = field(default=1e-5)
	per_device_train_batch_size: int = field(default=2)
	per_device_eval_batch_size: int = field(default=4)
	gradient_accumulation_steps: int = field(default=8)
	logging_strategy: str = field(default="steps")
	logging_steps: int = field(default=100)
	num_train_epochs: int = field(default=60)
	remove_unused_columns: bool = field(default=False)
	do_train: bool = field(default=True)
	do_eval: bool = field(default=True)
	eval_strategy: str = field(default="epoch")
	eval_steps: int = field(default=1)
	max_grad_norm: float = field(default=10)
	lr_scheduler_type: str = field(default="cosine_with_restarts")
	warmup_steps: int = field(default=1000)
	save_strategy: str = field(default="epoch")
	save_steps: int = field(default=1)
	save_total_limit: int = field(default=10)
	bf16: bool = field(default=True)
	dataloader_num_workers: int = field(default=8)
	dataloader_persistent_workers: bool = field(default=True)
	include_inputs_for_metrics: bool = field(default=False)
	eval_accumulation_steps: int = field(default=1)
	eval_on_start: bool = field(default=True)
	dataset_num_proc: int = field(default=8)
	packing: bool = field(default=False)
	max_seq_length: int = field(default=1024)  # 8192
	dataset_batch_size: int = field(default=256)
	neftune_noise_alpha: float = field(default=None)
	model_init_kwargs: dict = field(default=None)
	dataset_kwargs: dict = field(default=None)
	eval_packing: bool = field(default=None)
	num_of_sequences: int = field(default=1024)
	chars_per_token: float = field(default=3.6)


@dataclass
class ModelConfig(trl.ModelConfig):
	model_name_or_path: str = field(default=model_path)
	# model_revision="main"
	torch_dtype: str = field(default='bfloat16')
	# attn_implementation='flash_attention_2'
	use_peft: bool = field(default=True)
	lora_r: int = field(default=16)
	lora_alpha: int = field(default=32)
	lora_dropout: float = field(default=0.05)
	# layers_to_transform: List[int] = field(default_factory=lambda: list(range(14, 28))) # 效果差
	lora_target_modules: List[str] = field(
		default_factory=lambda: [
			"q_proj",
			"k_proj",
			"v_proj",
			"o_proj",
			"up_proj",
			"gate_proj",
			"down_proj",
		]
	)
	lora_modules_to_save: Optional[List[str]] = field(
		default=None,
		metadata={"help": "Model layers to unfreeze & train"},
	)
	lora_task_type: str = field(default="SEQ_CLS")
	load_in_8bit: bool = field(default=False)
	load_in_4bit: bool = field(default=False)