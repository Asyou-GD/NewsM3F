import sys
sys.path.append(sys.path[0] + '/../../../')
from mmhf.modeling_datasets.dataset_preprocess import format_chat_template, get_gt_label_for_seq_classification
from functools import partial
import mmengine
import os.path
import torch
import datasets
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer, PreTrainedModel, \
	BitsAndBytesConfig, AutoConfig
import time


def create_input(dataset_path, task_name='标题党', tokenizer=None, max_seq_length=8192, padding='max_length', device='cuda:0'):
	from mmhf.modeling_datasets.dataset_preprocess import apply_chat_template_for_seq_classification, seq_tokenizing_for_seq_classification
	# dataset
	raw_datasets = datasets.load_from_disk(dataset_path+'/test')
	map_gt_label_for_seq_classification = partial(get_gt_label_for_seq_classification, task_name=task_name)
	eval_dataset = raw_datasets.map(map_gt_label_for_seq_classification, num_proc=8)
	eval_dataset = eval_dataset.filter(lambda x: x['labels'] >= 0, num_proc=8)
	eval_dataset = eval_dataset.select(range(4))
	mmengine.dump(eval_dataset[0], 'work_dirs/eval_dataset_0.json', indent=4)
	print('eval_dataset: ', eval_dataset[0])

	apply_chat_template_for_seq_classification_ = partial(apply_chat_template_for_seq_classification, task_name=task_name)
	eval_dataset = eval_dataset.map(apply_chat_template_for_seq_classification_)
	seq_tokenizing_for_seq_classification_ = partial(
		seq_tokenizing_for_seq_classification,
		tokenizer=tokenizer,
		messages_field='messages',
		formatting_func=format_chat_template,
		add_special_tokens=False,
		max_seq_length=max_seq_length,
		padding=padding,
		return_tensors='pt'
	)

	signature_columns = ["input_ids", "attention_mask"]
	extra_columns = list(set(eval_dataset.column_names) - set(signature_columns))

	tokenized_dataset = eval_dataset.map(
		seq_tokenizing_for_seq_classification_,
		batched=True,
		remove_columns=extra_columns,
		batch_size=10
	)

	inputs_data = {}
	for i, item in enumerate(tokenized_dataset):
		for k, v in item.items():
			if k not in inputs_data:
				inputs_data[k] = []
			inputs_data[k].append(v)
	for k, v in inputs_data.items():
		inputs_data[k] = torch.tensor(v).to(device)
	return inputs_data


def benchmark(model, inputs_data, tokenizer, model_name_or_path=None):
	# model to fp32
	model = model.to(torch.float32)
	time_start = time.time()
	with torch.no_grad():
		results = model(**inputs_data)
	probs = torch.softmax(results.logits, dim=-1)
	print('original result: ', probs.cpu().numpy())
	print(f'pytorch fp32 model forward time: {time.time() - time_start:.4f}s')

	# model to bf16
	bf16_model = model.to(torch.bfloat16)
	time_start = time.time()
	with torch.no_grad():
		results = bf16_model(**inputs_data)
	probs = torch.softmax(results.logits.float(), dim=-1)
	print('original result: ', probs.cpu().numpy())
	print(f'pytorch bf16 model forward time: {time.time() - time_start:.4f}s')

	# model to fp16
	fp16_model = model.to(torch.float16)
	time_start = time.time()
	with torch.no_grad():
		results = fp16_model(**inputs_data)
	probs = torch.softmax(results.logits.float(), dim=-1)
	print('original result: ', probs.cpu().numpy())
	print(f'pytorch fp16 model forward time: {time.time() - time_start:.4f}s')

	quant_config = BitsAndBytesConfig(load_in_8bit=True)
	bit8_model = AutoModelForSequenceClassification.from_pretrained(
		model_name_or_path,
		quantization_config=quant_config,
		pad_token_id=tokenizer.pad_token_id,  # '<|im_end|>'
	)
	time_start = time.time()
	with torch.no_grad():
		results = bit8_model(**inputs_data)
	probs = torch.softmax(results.logits.float(), dim=-1)
	print('original result: ', probs.cpu().numpy())
	print(f'pytorch 8bit model forward time: {time.time() - time_start:.4f}s')


def merge_and_test(
		dataset_path,
		task_name,
		checkpoint_path,
		output_path,
		device='cuda:0',
		max_seq_length=4096,
		padding='max_length',
		is_save_bf16=False
):
	mmengine.mkdir_or_exist(output_path)

	# get tokenizer and model
	config = PeftConfig.from_pretrained(checkpoint_path)
	tokenizer = AutoTokenizer.from_pretrained(
		config.base_model_name_or_path,
		trust_remote_code=True,
		use_fast=True,
		return_tensors='pt'
	)
	model = AutoModelForSequenceClassification.from_pretrained(
		config.base_model_name_or_path,
		pad_token_id=tokenizer.pad_token_id,  # '<|im_end|>'
	)
	lora_model = PeftModel.from_pretrained(model, checkpoint_path)
	lora_model.to(device)
	lora_model.eval()
	torch_model = lora_model.merge_and_unload()
	# get convert data
	inputs_data = create_input(
		dataset_path,
		task_name=task_name,
		tokenizer=tokenizer,
		max_seq_length=max_seq_length,
		padding=padding,
		device=device
	)

	signature_columns = ["input_ids", "attention_mask"]
	print('signature_columns: ', signature_columns)

	if is_save_bf16:
		torch_model = torch_model.to(torch.bfloat16)
	torch_model.save_pretrained(output_path)
	tokenizer.save_pretrained(output_path)
	print(f"Model saved to {output_path}")

	benchmark(torch_model, inputs_data, tokenizer, model_name_or_path=output_path)




if __name__ == '__main__':
	qs_cloud = 'search01'  # 'search01' or 'nlp-ali' or 'nj-larc'
	time_data = '2024-08-16'
	task_name = '表意不规范'  # '标题党' or '表意不规范' or '夸大宣传'
	dataset_name = 'trainval_for_text_model'
	work_dir = f'work_dirs/quality/Qwen2-1.5B_biaoyi_{time_data}'

	ckpt_name = 'checkpoint-6592'
	upload_repository = 'contentproj-noteordermisrepresentation-v1-l20-torch'

	is_upload = True
	is_save_bf16 = True
	device = 'cuda:0'
	checkpoint_path = f'{work_dir}/{ckpt_name}'
	output_path = work_dir + f'_{ckpt_name}_torch_deploy'

	dataset_path = f'/mnt/{qs_cloud}/dataset/cky_data/notemarket/note_order_trainvaltest_data_df_{time_data.replace("-", "")}'
	dataset_path = dataset_path + f'/{dataset_name}'

	if is_upload and os.path.exists(output_path):
		cmd = f'python -m redserving_toolkit.transfer -t content_quality -s {upload_repository} -m classifier -f {output_path}'
		print(cmd)
		os.system(cmd)
		exit(0)
	else:
		merge_and_test(
			dataset_path,
			task_name,
			checkpoint_path,
			output_path,
			device='cuda:0',
			max_seq_length=4096,
			padding='longest',
			is_save_bf16=is_save_bf16
		)



'''
tensor([[0.0914, 0.9086],
        [0.9908, 0.0092],
        [0.9898, 0.0102],
        [0.9458, 0.0542]])

[0.09297106 0.9070289 ]
 [0.989175   0.01082494]
 [0.9897305  0.01026956]
 [0.94431746 0.0556825 ]
 
 [0.09138211 0.908618  ]
 [0.9908035  0.00919655]
 [0.9898303  0.01016966]
 [0.9458012  0.05419873]
 
 [[0.0921962  0.9078038 ]
 [0.9894907  0.01050938]
 [0.99002504 0.00997492]
 [0.94458723 0.0554128 ]]
 
 [[0.09268778 0.9073123 ]
 [0.98792297 0.01207706]
 [0.97529614 0.02470387]
 [0.8906752  0.10932483]]
'''