import json
import os
import cv2
import mmcv
import traceback
import numpy as np
import traceback
import redis
import copy
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Tuple, List

# thrift rpc 输入/输出结构体
from infra.rpc.base.ttypes import Result
from aimedia.modelinterface.ttypes import ResourceResponse, ResourceResult, FeatureType, ModelStatus, ModelResultInfo, LabelResponse, LabelInfo

# redserving框架默认基类
from base_inference import BaseInference
from red_serving_common import GetLogger  # 框架提供的标准logger, 要求版本 >= v1.X

# 自定义预处理
from param_config import InferenceCfg
from utils import apply_chat_template_for_seq_classification, seq_tokenizing_for_seq_classification, format_chat_template
from svc_downloader import parse_svc_and_download

logger = GetLogger()


class Classifier(BaseInference):
	_IMG_URL = 'img_url'
	_IMG_URLS = 'img_urls'
	_NOTE_ID = 'note_id'
	_USER_ID = 'user_id'

	def __init__(self):
		super().__init__()
		self.cfg = InferenceCfg()
		self.img_mean = np.array(self.cfg.img_mean, dtype=np.float32).reshape(1, 1, 3)
		self.img_std = np.array(self.cfg.img_std, dtype=np.float32).reshape(1, 1, 3)
		self.empty_img = np.ones((self.cfg.img_size, self.cfg.img_size, 3), dtype=np.uint8) * self.img_mean
		# download model
		parse_svc_and_download('./svc.yaml')

		model_download_info = json.loads(os.getenv('MODEL_DOWNLOADS', None))
		if model_download_info is None:
			raise ValueError('The environment variable MODEL_DOWNLOAD_INFO is not set correctly.')
		model_download_info = model_download_info.get('classifier', None)
		self.model_name_or_path = model_download_info['path'] + '/' + model_download_info['remote'] + '/' + model_download_info['version']
		self.cfg.model_name_or_path = self.model_name_or_path

		self.tokenizer = AutoTokenizer.from_pretrained(
			self.model_name_or_path,
			trust_remote_code=True,
			use_fast=True,
		)
		if self.cfg.torch_dtype == 'bf16':
			torch_dtype = torch.bfloat16
		elif self.cfg.torch_dtype == 'fp16':
			torch_dtype = torch.float16
		elif self.cfg.torch_dtype == 'fp32':
			torch_dtype = torch.float32
		else:
			logger.error(f"Unsupported torch_dtype: {self.cfg.torch_dtype}")
			raise ValueError(f"Unsupported torch_dtype: {self.cfg.torch_dtype}")

		logger.info(f"model_name_or_path: {self.model_name_or_path}")
		self.model = AutoModelForSequenceClassification.from_pretrained(
			self.model_name_or_path,
			pad_token_id=self.tokenizer.pad_token_id,
			device_map=self.cfg.device
		)
		self.model = self.model.to(torch_dtype)
		self.model.eval()
		logger.info(f"model loaded finished: {self.model_name_or_path}")

		# model parameters
		self.labels_cfg = {
			"num_labels": self.model.config.num_labels,
			'class_names': self.cfg.class_names,
			'cls2idx': {v: i for i, v in enumerate(self.cfg.class_names)}
		}

		# image download source & image redkv
		self.download_source = ['failed', 'binary', 'redkv', 'url']
		self.img_kv_client = redis.Redis(host=self.cfg.redkv_host, port=self.cfg.redkv_host)

	## @BaseInference.reset_helper 在每个需要被RedServing框架调用的函数前必须有；
	@BaseInference.reset_helper
	def pre_process(self, request):
		# trace_id 是每个请求所独有的，可用于打印log以帮助线上追踪/故障定位
		trace_id = self.req_helper.get_trace_id()
		try:
			## 通过内置的self.get_rpc_resource_request获取原始RPC请求信息；
			resource_req = self.get_rpc_resource_request()

			# 获取请求参数
			inputs_data = self.reorganize_request_data(resource_req.requestData)
			self.res_helper.add_user_cache('info', {
				self._IMG_URLS: inputs_data[self._IMG_URLS],
				self._NOTE_ID: inputs_data[self._NOTE_ID],
				self._USER_ID: inputs_data[self._USER_ID]
			})

			# ## 最后需要 生成模型推理需要的输入，目前只支持numpy.array格式；
			# imgs, _, _, _ = self.get_imgbin(resource_req.featureData)
			# info['img_attn_masks'] = np.array([1] * len(imgs) + [0] * (self.num_imgs - len(imgs)), dtype=self.int_type).reshape(1, self.num_imgs)
			# info["inputs"] = self._preprocess_image(imgs)

			# 包装成推理模型的输入tensor
			# for idx, field in enumerate(input_names):
			# 	self.res_helper.add_infer_tensor(info[field], name=f"INPUT__{idx}")
			with torch.no_grad():
				results = self.model(input_ids=inputs_data['input_ids'], attention_mask=inputs_data['attention_mask'])
			probs = torch.softmax(results.logits.float(), dim=-1)
			probs = probs.detach().cpu().numpy()
			rpc_response = self._post_process(probs)
		except:
			error_info = f'[{trace_id}] Error in pre_process: {traceback.format_exc()}'
			logger.error(error_info)
			rpc_response = ResourceResponse(Result(success=False, code=103, message=error_info))
			self.res_helper.set_rpc_response(rpc_response)
			return self.res_helper.generate_response_and_return(terminate=True)  # 如果设置terminate=True, 将直接结束整个处理流程；
		self.res_helper.set_rpc_response(rpc_response)
		return self.res_helper.generate_response_and_return()

	## @BaseInference.reset_helper 在每个需要被RedServing框架调用的函数前必须有；
	@BaseInference.reset_helper
	def post_process(self, request):
		trace_id = self.req_helper.get_trace_id()
		try:
			probs = self.req_helper.get_input_tensors()[0]
			info = self.req_helper.get_user_cache('info')
			rpc_response = self._post_process(probs, info=info)
		except:
			error_info = f'[{trace_id}] Error in post_process: {traceback.format_exc()}'
			logger.error(error_info)
			rpc_response = ResourceResponse(Result(success=False, code=100, message=error_info))

		self.res_helper.set_rpc_response(rpc_response)
		return self.res_helper.generate_response_and_return()

	# ---------自定义辅助函数-------------
	def _preprocess_image(self, img_: List[np.ndarray]):
		img_ = img_ + [self.mean_bgr_image] * (self.num_imgs - len(img_))

		img_new = []
		for im in img_:
			im = copy.deepcopy(im.astype(np.float32))
			im = mmcv.imresize(im, self.size, interpolation='bicubic', return_scale=False, backend='cv2')
			im = mmcv.imnormalize(im, self.mean, self.std, self.to_rgb)
			im = im.transpose(2, 0, 1)  # [c, h, w]
			img_new.append(im)

		img_new = np.stack(img_new, axis=0)[np.newaxis, ...]  # [1, 9, c, h, w]
		return img_new.astype(self.float_type)

	def _post_process(self, probs):
		result = ResourceResult()
		model_result = ModelResultInfo()
		label_response = LabelResponse()

		label_infos = []
		cls2score = dict()

		max_score = -1
		max_label = 'None'
		for idx, score in enumerate(probs[0]):
			score = round(float(score), 4)
			cls_name = self.labels_cfg['class_names'][idx]
			label_info = LabelInfo(labelID=idx, labelName=cls_name, score=score)
			cls2score[cls_name] = score
			label_infos.append(label_info)
			if score > max_score:
				max_score = score
				max_label = cls_name

		result.modelStatus = ModelStatus.SUCCEED_NORMAL
		label_response.status = ModelStatus.SUCCEED_NORMAL
		model_result.label = max_label
		model_result.score = max_score

		label_response.labelInfos = label_infos
		model_result.labelResponse = label_response
		result.modelResult = model_result
		cv_response = ResourceResponse(result=Result(success=True, code=0, message="None"), resourceResult=result)

		return cv_response

	def get_imgbin(self, featureData) -> Tuple[list, list, bool]:

		def get_imgbin_by_idx(idx: int):
			# 二进制读图
			if idx < num_binary:
				img_bin = binary_data.binaryListValue[idx]
				if img_bin is not None: return img_bin, 1  # is_download == 1 二进制读图成功

			# redkv 读图
			if idx < num_cache:
				cache_key = cache_data.stringListValue[idx]
				img_bin = self.img_kv_client.get(cache_key)
				if img_bin is not None: return img_bin, 2  # is_download == 2 redkv 读图成功

			# 临时下图
			if idx < num_urls:
				url = imgurl_data.stringListValue[idx]
				if not url.startswith('http'):
					url = 'https://xhsci-10008268.cos.ap-shanghai.myqcloud.com/' + url
				elif url.startswith('http://ci.xiaohongshu.com/'):
					url.replace('http://ci.xiaohongshu.com/', 'https://xhsci-10008268.cos.ap-shanghai.myqcloud.com/')
				url = url.rsplit('?')[0] + '?imageView2/2/w/1080/h/1080/format/webp/q/75'
				resp = requests.get(url, timeout=2)
				if resp.ok:
					img_bin = resp.content
					return img_bin, 3  # is_download == 3 临时下图成功

			return None, 0  # is_download == 0 下载图片失败

		binary_data = featureData.get('binaryDatas', None)
		cache_data = featureData.get('resourceCacheKeys', None)
		imgurl_data = featureData.get('imageUrls', None)

		num_binary = len(
			binary_data.binaryListValue) if binary_data is not None and binary_data.type == FeatureType.BINARY_LIST else 0
		num_cache = len(
			cache_data.stringListValue) if cache_data is not None and cache_data.type == FeatureType.STRING_LIST else 0
		num_urls = len(
			imgurl_data.stringListValue) if imgurl_data is not None and imgurl_data.type == FeatureType.STRING_LIST else 0

		num_images = max(num_binary, num_cache, num_urls)
		num_frames = min(self.num_imgs, num_images)
		frame_indices = list(range(num_frames))

		img_info = [get_imgbin_by_idx(i) for i in frame_indices]
		imgs_bin, is_download = list(zip(*img_info))

		imgs = [cv2.imdecode(np.frombuffer(_, np.uint8), cv2.IMREAD_COLOR) if _ is not None else None for _ in imgs_bin]
		return imgs, imgs_bin, is_download, all(is_download)

	def reorganize_request_data(self, requestData: Optional[dict]) -> dict:

		return_data_dict = {}
		required_fields = self.cfg.required_fields
		if self._IMG_URL in requestData and self._IMG_URLS not in requestData:
			requestData[self._IMG_URLS] = [requestData[self._IMG_URL]]

		for field in required_fields:
			if field not in requestData:
				raise ValueError(f'{field} is required in requestData')
			return_data_dict[field] = requestData[field]
		return_data_dict[self._NOTE_ID] = requestData.get(self._NOTE_ID, None)
		return_data_dict[self._USER_ID] = requestData.get(self._USER_ID, None)

		tmp_data_dict = apply_chat_template_for_seq_classification(return_data_dict, task_name=self.cfg.task_name)

		tmp_data_dict = seq_tokenizing_for_seq_classification(
			tmp_data_dict,
			tokenizer=self.tokenizer,
			messages_field='messages',
			formatting_func=format_chat_template,
			add_special_tokens=False,
			max_seq_length=self.cfg.max_seq_length,
			padding=self.cfg.padding,
		)
		signature_columns = ["input_ids", "attention_mask"]
		for key in signature_columns:
			# return_data_dict[key] = tmp_data_dict[key].cpu().numpy().astype(self.cfg.int_type)
			return_data_dict[key] = tmp_data_dict[key].to(self.cfg.device)
		return return_data_dict
