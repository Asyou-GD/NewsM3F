import pathlib
import sys
from typing import Optional
import numpy as np

from mmhf.deploy.convert_model.torch_allocator import TorchAllocator

# from optimum.exporters.onnx import OnnxConfig
# from optimum.exporters.onnx.model_configs import Qwen2OnnxConfig
sys.path.append(sys.path[0] + '/../../')
from mmhf.deploy.convert_model.wrap_model import WrappedModel, WrappedModelFp16

from mmhf.datasets.dataset_preprocess import format_chat_template, get_gt_label_for_seq_classification
from functools import partial
import mmengine
import os.path
from copy import deepcopy
import torch
import datasets
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer, PreTrainedModel, \
	BitsAndBytesConfig, AutoConfig
import onnxruntime as ort
import tensorrt as trt
import onnx
import time


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
	if dtype == trt.bool:
		return torch.bool
	elif dtype == trt.int8:
		return torch.int8
	elif dtype == trt.int32:
		return torch.int32
	elif dtype == trt.float16:
		return torch.float16
	elif dtype == trt.float32:
		return torch.float32
	else:
		raise TypeError(f'{dtype} is not supported by torch')


def torch_device_from_trt(device: trt.TensorLocation):
	"""Convert pytorch device to TensorRT device.

    Args:
        device (trt.TensorLocation): The device in tensorrt.
    Returns:
        torch.device: The corresponding device in torch.
    """
	if device == trt.TensorLocation.DEVICE:
		return torch.device('cuda')
	elif device == trt.TensorLocation.HOST:
		return torch.device('cpu')
	else:
		return TypeError(f'{device} is not supported by torch')


def create_input(dataset_path, task_name='标题党', tokenizer=None, max_seq_length=8192, padding='max_length', device='cuda:0'):
	from mmhf.datasets.dataset_preprocess import apply_chat_template_for_seq_classification, seq_tokenizing_for_seq_classification
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


def test_optimum_nvidia(model_name_or_path, inputs_data, tokenizer):
	from optimum.nvidia.pipelines import pipeline
	from optimum.nvidia import AutoModelForSequenceClassification

	model = AutoModelForSequenceClassification.from_pretrained(
		model_name_or_path,
		bf16=True,
		max_prompt_length=8192,
		max_batch_size=8,
		pad_token_id=tokenizer.pad_token_id,
		num_labels=2,
	)

	outputs = model(**inputs_data)
	print(outputs)


def benchmark(model_name_or_path, inputs_data, tokenizer):

	torch_model = AutoModelForSequenceClassification.from_pretrained(
		model_name_or_path,
		pad_token_id=tokenizer.pad_token_id,
		return_dict=False,
		torch_dtype=torch.bfloat16,
		device_map="auto"
	)
	torch_model = WrappedModel(torch_model)
	time_start = time.time()
	with torch.no_grad():
		results = torch_model(**inputs_data)
	print(len(results))
	print('original result: ', results.float().detach().cpu().numpy())
	print(f'pytorch bf16 model forward time: {time.time() - time_start:.4f}s')

	quant_config = BitsAndBytesConfig(load_in_8bit=True)
	torch_model = AutoModelForSequenceClassification.from_pretrained(
		model_name_or_path,
		pad_token_id=tokenizer.pad_token_id,
		return_dict=False,
		quantization_config=quant_config,
		device_map="auto"
	)
	torch_model = WrappedModel(torch_model)
	time_start = time.time()
	with torch.no_grad():
		results = torch_model(**inputs_data)
	print(len(results))
	print('original result: ', results.float().detach().cpu().numpy())
	print(f'pytorch 8bit model forward time: {time.time() - time_start:.4f}s')


def optimum_onnx_trt(model_name_or_path, inputs_data, tokenizer):
	import onnxruntime
	from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer, ORTOptimizer, AutoOptimizationConfig, ORTConfig
	from optimum.onnxruntime.configuration import AutoQuantizationConfig
	from transformers import AutoTokenizer, pipeline
	time_start = time.time()
	onnx_save_directory = model_name_or_path+'_onnx'
	trt_save_directory = model_name_or_path+'_trt'
	session_options = onnxruntime.SessionOptions()
	session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

	# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	#
	# ort_model = ORTModelForSequenceClassification.from_pretrained(
	# 	onnx_save_directory,
	# 	provider="TensorrtExecutionProvider",
	# 	session_options=session_options,
	# 	provider_options={
	# 		"trt_fp16_enable": False,
	# 		'trt_int8_enable': False,
	# 		"trt_engine_cache_enable": True,
	# 		"trt_engine_cache_path": trt_save_directory
	# 	},
	# 	export=True,
	# )
	# ort_model.save_pretrained(trt_save_directory)
	# tokenizer.save_pretrained(trt_save_directory)
	# print(f'Convert model successfully, time: {time.time() - time_start:.4f}s')

	ort_model = ORTModelForSequenceClassification.from_pretrained(
		onnx_save_directory,
		provider="TensorrtExecutionProvider",
		session_options=session_options,
		provider_options={
			"trt_fp16_enable": False,
			'trt_int8_enable': False,
			"trt_engine_cache_enable": True,
			"trt_engine_cache_path": trt_save_directory
		},
	)

	time_start = time.time()
	results = ort_model(**inputs_data)
	results = torch.softmax(results[0], dim=-1)
	print('onnx result: ', results)
	print(f'onnx model predict time: {time.time() - time_start:.4f}s')


def optimum_onnx_cuda(model_name_or_path, inputs_data, tokenizer):
	import onnxruntime
	from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer, ORTOptimizer, AutoOptimizationConfig, OptimizationConfig
	from optimum.onnxruntime.configuration import AutoQuantizationConfig
	from transformers import AutoTokenizer, pipeline
	time_start = time.time()
	onnx_save_directory = model_name_or_path+'_onnx'
	session_options = onnxruntime.SessionOptions()
	session_options.log_severity_level = 0

	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

	ort_model = ORTModelForSequenceClassification.from_pretrained(
		model_name_or_path,
		provider="CUDAExecutionProvider",
		session_options=session_options,
		use_io_binding=True,
		export=True
	)

	# optimization_config = OptimizationConfig(
	# 	optimization_level=0,
	# 	enable_transformers_specific_optimizations=True,
	# 	optimize_for_gpu=True,
	# 	fp16=True
	# )
	# optimizer = ORTOptimizer.from_pretrained(ort_model)
	# optimizer.optimize(save_dir=onnx_save_directory, optimization_config=optimization_config)

	ort_model.save_pretrained(onnx_save_directory)
	tokenizer.save_pretrained(onnx_save_directory)
	print(f'Convert model successfully, time: {time.time() - time_start:.4f}s')

	ort_model = ORTModelForSequenceClassification.from_pretrained(
		onnx_save_directory,
		provider="CUDAExecutionProvider",
		session_options=session_options,
		use_io_binding=True,
	)

	time_start = time.time()
	results = ort_model(**inputs_data)
	results = torch.softmax(results[0], dim=-1)
	print('onnx result: ', results)
	print(f'onnx model predict time: {time.time() - time_start:.4f}s')


def optimum_export_onnx_cuda(model_name_or_path, inputs_data, tokenizer):
	import onnxruntime
	from optimum.exporters.onnx import main_export, export
	from optimum.onnxruntime import ORTModelForSequenceClassification
	from transformers import AutoTokenizer
	time_start = time.time()
	onnx_save_directory = model_name_or_path+'_onnx'

	# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	# base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
	onnx_config = dict(
		model=Qwen2OnnxConfig(
				config=AutoConfig.from_pretrained(model_name_or_path),
				task='text-classification',
				int_dtype='int32',
				float_dtype='bf16')
	)
	main_export(
		model_name_or_path,
		onnx_save_directory,
		task='text-classification',
		device='cuda',
		opset=17,
		# optimize='O2',  # None
		monolith=True,
		no_post_process=True,
		# dtype='bf16',
		# no_post_process=True,
		# model_kwargs={"output_attentions": True},
		# custom_onnx_configs=onnx_config
	)
	# export(
	# 	base_model,
	# 	onnx_config,
	# 	onnx_save_directory,
	# 	device='cuda',
	# 	opset=17,
	# 	# optimize='O2',  # None
	# 	# monolith=True,
	# 	# no_post_process=True,
	# 	# no_post_process=True,
	# 	# model_kwargs={"output_attentions": True},
	# 	# custom_onnx_configs=custom_onnx_configs
	# )
	print(f'Convert model successfully, time: {time.time() - time_start:.4f}s')

	ort_model = ORTModelForSequenceClassification.from_pretrained(
		onnx_save_directory,
		provider="CUDAExecutionProvider",
		use_io_binding=True,
	)

	time_start = time.time()
	results = ort_model(**inputs_data)
	results = torch.softmax(results[0], dim=-1)
	print('onnx result: ', results)
	print(f'onnx model predict time: {time.time() - time_start:.4f}s')


def convert_torch_to_onnx_to_trt(
		dataset_path,
		task_name,
		checkpoint_path,
		output_onnx_path,
		output_trt_path,
		is_onnx_convert=True,
		is_onnx_predict=True,
		is_trt_convert=True,
		is_trt_predict=True,
		is_trt_fp16=True,
		device='cuda:0',
		verbose=False,
		max_seq_length=4096,
		padding='longest',
):
	mmengine.mkdir_or_exist(os.path.dirname(output_onnx_path))
	mmengine.mkdir_or_exist(os.path.dirname(output_trt_path))

	tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True, use_fast=True)

	model = AutoModelForSequenceClassification.from_pretrained(
		checkpoint_path,
		pad_token_id=tokenizer.pad_token_id,  # '<|im_end|>'
		# torch_dtype=torch.bfloat16,
	)
	model = model.to(device)
	model.eval()
	# torch_model = model.to(torch.bfloat16)
	if is_trt_fp16:
		torch_model = WrappedModelFp16(model)
	else:
		torch_model = WrappedModel(model)
	# get convert data
	inputs_data = create_input(dataset_path, task_name=task_name, tokenizer=tokenizer, max_seq_length=max_seq_length, padding=padding)

	# # test torch-tensorrt
	# import torch_tensorrt
	# optimized_model = torch.compile(torch_model, backend="tensorrt")
	# optimized_model(**inputs_data)
	# optimized_model(**inputs_data)

	keys_sorted = ["input_ids", "attention_mask"]
	print('keys_sorted: ', keys_sorted)
	input_names = [f'INPUT__{i}' for i in range(len(keys_sorted))]

	# 只输出分类结果
	output_names = ['OUTPUT__0']
	# 输出分类结果和特征
	# output_names = ['OUTPUT__0', 'OUTPUT__1']

	time_start = time.time()
	with torch.no_grad():
		results = model(**inputs_data)
		results = torch.softmax(results.logits, dim=-1)
	print(f'torch model forward time: {time.time() - time_start:.4f}s')
	print('original result: ', results.detach().cpu().numpy())

	# benchmark('work_dirs/model_deploy/Qwen2-1.5B_title_2024-08-12/checkpoint-3570', inputs_data, tokenizer)
	# optimum_export_onnx_cuda('work_dirs/model_deploy/Qwen2-1.5B_title_2024-08-12/checkpoint-3570', inputs_data, tokenizer)
	# optimum_onnx_trt('work_dirs/model_deploy/Qwen2-1.5B_title_2024-08-12/checkpoint-3570', inputs_data, tokenizer)
	# results = test_optimum_nvidia('work_dirs/model_deploy/Qwen2-1.5B_title_2024-08-12/checkpoint-3570', inputs_data, tokenizer)

	if is_onnx_convert:
		with torch.no_grad():
			input_tuple_order = []
			for k in keys_sorted:
				input_tuple_order.append(inputs_data[k].cpu())
			if is_trt_fp16:
				inputs_embeds = torch_model.get_input_embeddings(input_tuple_order[0].to(device))
				input_tuple_order[0] = inputs_embeds.cpu()
			input_tuple_order = tuple(input_tuple_order)
			time_start = time.time()
			dynamic_axes = {x: {0: 'batch_size'} for x in input_names + output_names}
			torch.onnx.export(
				torch_model.cpu(),
				input_tuple_order,
				output_onnx_path,
				export_params=True,
				verbose=verbose,
				input_names=input_names,
				output_names=output_names,
				opset_version=17,
				dynamic_axes=dynamic_axes,
			)
			print(f'onnx model convert time: {time.time() - time_start:.4f}s')
			print(f'Convert to onnx model successfully, save to {output_onnx_path}')
			tokenizer.save_pretrained(os.path.dirname(output_onnx_path))
			print(f'Save tokenizer successfully, save to {os.path.dirname(output_onnx_path)}')

	if is_onnx_predict:
		sess_options = ort.SessionOptions()
		ort_session = ort.InferenceSession(output_onnx_path, sess_options, providers=['CUDAExecutionProvider'])
		input_dict_order = {}
		for i, key in enumerate(keys_sorted):
			input_dict_order[input_names[i]] = inputs_data[key].cpu().numpy()
		if is_trt_fp16:
			input_dict_order[input_names[0]] = torch_model.cuda().get_input_embeddings(inputs_data['input_ids'].to(device)).detach().cpu().numpy()
		time_start = time.time()
		outputs = ort_session.run(output_names, input_dict_order)
		print(f'onnx model predict time: {time.time() - time_start:.4f}s')
		if is_trt_fp16:
			outputs = torch_model.get_scores(outputs[0], inputs_data['input_ids'])
			outputs = (outputs, )
		print('onnx result: ', outputs[0])

	if is_trt_convert:
		inputs = {}
		for i, k in enumerate(keys_sorted):
			inputs[f'INPUT__{i}'] = inputs_data[k]

		if is_trt_fp16:
			inputs['INPUT__0'] = torch_model.get_input_embeddings(inputs_data['input_ids'])

		input_shapes = {}
		bs_min = 1
		bs_opt = 2
		bs_max = 4
		for k, v in inputs.items():
			input_shapes[k] = {
				'min_shape': [bs_min] + list(v.shape)[1:],
				'opt_shape': [bs_opt] + list(v.shape)[1:],
				'max_shape': [bs_max] + list(v.shape)[1:]
			}
		time_start = time.time()
		# 创建TensorRT日志器
		TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

		# 创建TensorRT引擎构建器和网络定义
		builder = trt.Builder(TRT_LOGGER)
		network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

		# 解析ONNX模型
		parser = trt.OnnxParser(network, TRT_LOGGER)

		if isinstance(output_onnx_path, str):
			parse_valid = parser.parse_from_file(output_onnx_path)
		elif isinstance(output_onnx_path, onnx.ModelProto):
			parse_valid = parser.parse(output_onnx_path.SerializeToString())
		else:
			raise TypeError('Unsupported onnx model type!')

		if not parse_valid:
			error_msgs = ''
			for error in range(parser.num_errors):
				error_msgs += f'{parser.get_error(error)}\n'
			raise RuntimeError(f'Failed to parse onnx, {error_msgs}, onnx model: {output_onnx_path}')

		# 配置TensorRT引擎
		config = builder.create_builder_config()
		max_workspace_size = 30 << 30  # 20GB
		if hasattr(config, 'set_memory_pool_limit'):
			config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
		else:
			config.max_workspace_size = max_workspace_size

		profile = builder.create_optimization_profile()
		for k, v in input_shapes.items():
			profile.set_shape(k, v['min_shape'], v['opt_shape'], v['max_shape'])

		if config.add_optimization_profile(profile) < 0:
			print(f'Invalid optimization profile {profile}.')

		if is_trt_fp16:
			config.set_flag(trt.BuilderFlag.FP16)

		if hasattr(builder, 'build_serialized_network'):
			engine = builder.build_serialized_network(network, config)
		else:
			engine = builder.build_engine(network, config)
		if isinstance(output_trt_path, str):
			with open(output_trt_path, 'wb') as f:
				if isinstance(engine, trt.ICudaEngine):
					engine = engine.serialize()
				f.write(bytearray(engine))
			print(f'Convert to TensorRT model successfully, save to {output_trt_path}')
			print(f'tensorrt model convert time: {time.time() - time_start:.4f}s')
			tokenizer.save_pretrained(os.path.dirname(output_trt_path))
			print(f'Save tokenizer successfully, save to {os.path.dirname(output_trt_path)}')

	if is_trt_predict:
		# 加载序列化的TensorRT引擎
		allocator = TorchAllocator(0)
		with trt.Logger() as logger, trt.Runtime(logger) as runtime:
			if allocator is not None:
				runtime.gpu_allocator = allocator
			with open(output_trt_path, mode='rb') as f:
				engine_bytes = f.read()
			trt.init_libnvinfer_plugins(logger, namespace='')
			engine = runtime.deserialize_cuda_engine(engine_bytes)

		context = engine.create_execution_context()

		if hasattr(context, 'temporary_allocator'):
			context.temporary_allocator = allocator

		time_start = time.time()
		inputs = {}
		for i, k in enumerate(keys_sorted):
			inputs[f'INPUT__{i}'] = inputs_data[k].cuda().contiguous()
			if inputs[f'INPUT__{i}'].dtype == torch.long:
				inputs[f'INPUT__{i}'] = inputs[f'INPUT__{i}'].int()
		if is_trt_fp16:
			inputs['INPUT__0'] = torch_model.cuda().get_input_embeddings(inputs_data['input_ids']).cuda().contiguous()
		bindings = [None] * engine.num_io_tensors
		outputs = {}
		for i in range(engine.num_io_tensors):
			name = engine.get_tensor_name(i)
			shape = context.get_tensor_shape(name)
			mode = engine.get_tensor_mode(name)
			dtype = torch_dtype_from_trt(engine.get_tensor_dtype(name))
			device = torch.device('cuda')
			print(f'name: {name}, shape: {shape}, mode: {mode}, dtype: {dtype}, device: {device}')
			if mode == trt.TensorIOMode.INPUT:
				assert name in inputs
				input_tensor = inputs[name]
				assert 'cuda' in input_tensor.device.type
				context.set_input_shape(name, tuple(input_tensor.shape))
				bindings[i] = input_tensor.data_ptr()
			else:
				output = torch.empty(tuple(shape), dtype=dtype, device=device)
				outputs[name] = output
				bindings[i] = output.data_ptr()

		context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
		# outputs = get_score(outputs['OUTPUT__0'], inputs_data['input_ids'], tokenizer.pad_token_id)
		print(f'tensorrt model predict time: {time.time() - time_start:.4f}s')

		if is_trt_fp16:
			outputs['OUTPUT__0'] = torch_model.get_scores(outputs['OUTPUT__0'], inputs_data['input_ids'])
		print('trt result: ', outputs['OUTPUT__0'].detach().cpu().numpy())


if __name__ == '__main__':
	qs_cloud = 'search01'  # 'search01' or 'nlp-ali' or 'nj-larc'
	time_data = '2024-08-16'
	task_name = '标题党'  # '标题党' or '表意不规范' or '夸大宣传'
	dataset_name = 'trainval_for_text_model'
	work_dir = f'work_dirs/quality/Qwen2-1.5B_title_{time_data}'
	ckpt_name = 'checkpoint-9587_nlp'
	upload_repository = 'contentproj-noteorderclickbait-test-l20-trt'
	torch_deploy_path = work_dir + f'_{ckpt_name}_torch_deploy'

	is_upload = False
	is_onnx_convert = False
	is_onnx_predict = True
	is_trt_convert = True
	is_trt_predict = True
	is_trt_fp16 = True
	device = 'cuda:0'
	checkpoint_path = f'{work_dir}/{ckpt_name}'
	output_onnx_path = work_dir+f'_{ckpt_name}_onnx_deploy/model.onnx'

	output_trt_path = work_dir+f'_{ckpt_name}_trt_deploy/model.plan'

	dataset_path = f'/mnt/{qs_cloud}/dataset/cky_data/notemarket/note_order_trainvaltest_data_df_{time_data.replace("-", "")}'
	dataset_path = dataset_path + f'/{dataset_name}'

	if is_upload and os.path.exists(output_trt_path):
		# cmd = f'mv {output_trt_path} {output_trt_path.replace(".engine", ".plan")}'
		# print(cmd)
		# os.system(cmd)
		cmd = f'python -m redserving_toolkit.transfer -t content_quality -s {upload_repository} -m classifier -f {os.path.dirname(output_trt_path)}'
		print(cmd)
		os.system(cmd)
		sys.exit(0)
	if is_onnx_convert or is_onnx_predict or is_trt_convert or is_trt_predict:
		convert_torch_to_onnx_to_trt(
			dataset_path,
			task_name,
			torch_deploy_path,
			output_onnx_path,
			output_trt_path,
			is_onnx_convert,
			is_onnx_predict,
			is_trt_convert,
			is_trt_predict,
			is_trt_fp16=is_trt_fp16,
			device=device,
			verbose=False,
			max_seq_length=2048,
			padding='max_length',
		)
