import os
from dataclasses import dataclass, field
from typing import Optional, List, Union

import mmengine
import transformers
from accelerate import InitProcessGroupKwargs
from transformers.trainer_pt_utils import AcceleratorConfig

qs_cloud = 'nlp-ali'  # 'search01' or 'nlp-ali' or 'nj-larc' or gy-1
time_data = '2024-09-08'
task_name = '标题党'  # '标题党' or '表意不规范' or '夸大宣传'
tmpl_name = 'title_LLMV0'
work_dir = f'work_dirs/quality/Qwen2-1.5B_{time_data}_{tmpl_name}'

# qs_cloud = 'nlp-ali'  # 'search01' or 'nlp-ali' or 'nj-larc'
# time_data = '2024-08-21'
# task_name = '夸大宣传'  # '标题党' or '表意不规范' or '夸大宣传'
# work_dir = f'work_dirs/quality/Qwen2-1.5B_xuanchuan_{time_data}'


# qs_cloud = 'search01'  # 'search01' or 'nlp-ali' or 'nj-larc'
# time_data = '2024-08-21'
# task_name = '表意不规范'  # '标题党' or '表意不规范' or '夸大宣传'
# tmpl_name = '表意不规范LLM'
# work_dir = f'work_dirs/quality/Qwen2-1.5B_biaoyi_{time_data}'

dataset_name = 'trainval_webdataset_text'
model_path = f'work_dirs/model_cache/Qwen2-1.5B-Instruct'
dataset_path = f'/mnt/{qs_cloud}/dataset/cky_data/notemarket'
dataset_path = dataset_path + f'/{dataset_name}'

qs_cloud = 'nlp-ali'  # 'search01' or 'nlp-ali' or 'nj-larc' or gy-1
time_data = '20240911'
task_name = '视频质量'  # '标题党' or '表意不规范' or '夸大宣传'
tmpl_name = 'video_LLMV0'
work_dir = f'work_dirs/quality/Qwen2-1.5B_{time_data}_{tmpl_name}_4096'
dataset_path = f'/mnt/{qs_cloud}/dataset/cky_data/videoquality'
dataset_path = dataset_path + f'/{dataset_name}'


os.environ["WANDB_API_KEY"] = "bd1b24e183d6fe3c14ecdbcc8d07c7dd0ec540e5"
os.environ['WANDB_DIR'] = f'{work_dir}/wandb'
mmengine.mkdir_or_exist(os.environ['WANDB_DIR'])
os.environ['WANDB_PROJECT'] = f'XHS'


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=model_path)


@dataclass
class DataArguments:
    dataset_name: str = field(default=dataset_path, metadata={"help": "Path to the training data."})
    dataset_train_split: str = field(default="train/*.tar", metadata={"help": "The name of the training split."})
    dataset_test_split: str = field(default="test/*/*.arrow", metadata={"help": "The name of the test split."})
    task_name: str = field(default=task_name)
    tmpl_name: str = field(default=tmpl_name)
    train_neg_num: int = field(default=146301)  # 训练集负类别数目
    train_pos_num: int = field(default=9725)  # 训练集正类别数目
    dataset_num_proc: int = field(default=1)
    negative_sample_keep_ratio: float = field(default=0.2)
    max_seq_length: int = field(
        default=8192,  # 纯文本任务使用4096, 6144, 8192
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    buffer_size: int = field(default=5000)  # 需要修改
    cal_on_precision: List[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    cal_on_types: List[int] = field(default_factory=lambda: [2])


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    use_lora: bool = True
    dispatch_batches: bool = field(default=False)

    # accelerator_config: dict = field(
    #     default_factory=lambda: {"kwargs_handlers": [InitProcessGroupKwargs(backend="hccl")]})

    label_names: List[str] = field(default_factory=lambda: ["labels"])
    output_dir: str = field(default=work_dir)
    run_name: str = field(default=work_dir.split('/')[-1])
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_All_R_at_P50")
    greater_is_better: bool = field(default=True)
    report_to: str = field(default="wandb")
    optim: str = field(default="adamw_torch")
    learning_rate: float = field(default=1e-4)
    ignore_data_skip: bool = field(default=False)

    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=16)
    log_level: str = field(default="debug")
    gradient_accumulation_steps: int = field(default=1)
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=50)
    num_train_epochs: int = field(default=20)
    remove_unused_columns: bool = field(default=True)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    max_grad_norm: float = field(default=30)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    warmup_steps: int = field(default=1000)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=10)
    eval_accumulation_steps: int = field(default=1)
    eval_on_start: bool = field(default=True)
    bf16: bool = field(default=True)
    dataloader_num_workers: int = field(default=16)
    dataloader_persistent_workers: bool = field(default=True)
    include_inputs_for_metrics: bool = field(default=False)
    neftune_noise_alpha: float = field(default=None)


@dataclass
class LoraArguments:
    r: int = field(default=16)  # 16:32/ 32:32
    lora_alpha: int = field(default=32)  # 32
    lora_dropout: float = field(default=0.05)  # 0.05
    target_modules: List[str] = field(
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
    bias: str = field(default='none')
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train"},
    )
    task_type: str = field(default="SEQ_CLS")
    q_lora: bool = field(default=False)
