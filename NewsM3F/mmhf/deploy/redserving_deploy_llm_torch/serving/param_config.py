from dataclasses import dataclass, field
from typing import Union

import numpy as np


chat_templates = {
	'标题党': "请帮我判断下述图文笔记的质量，输出0或1，其中0表示优质笔记，1表示营销笔记或者劣质笔记。下面是笔记内容：\n标题：{title}\n内容：{content}\n图片的OCR：{ocr}。",
	'夸大宣传': "请帮我判断下述图文笔记的质量，输出0或1，其中0表示优质笔记，1表示营销笔记或者劣质笔记。下面是笔记内容：\n标题：{title}\n内容：{content}\n图片的OCR：{ocr}。",
	'表意不规范': "请帮我判断下述图文笔记的质量，输出0或1，其中0表示优质笔记，1表示营销笔记或者劣质笔记。下面是笔记内容：\n标题：{title}\n内容：{content}\n图片的OCR：{ocr}。",
}


@dataclass
class InferenceCfg:
	img_size: int = field(default=512)
	img_mean: list = field(default_factory=lambda: [123.675, 116.28, 103.53])
	img_std: list = field(default_factory=lambda: [58.395, 57.12, 57.375])
	img_to_rgb: bool = field(default=True)
	num_imgs: int = field(default=5)
	# numpy array dtype
	int_type: np.dtype = field(default=np.int32)
	float_type: np.dtype = field(default=np.float32)
	model_name_or_path: str = field(default='bert-base-uncased')
	class_names: list = field(default_factory=lambda: ['good', 'bad'])
	redkv_host: str = field(default='localhost')
	redkv_port: int = field(default=20320)
	task_name: str = field(default='标题党')
	max_seq_length: int = field(default=4096)
	required_fields: list = field(default_factory=lambda: ['title', 'content', 'ocr', 'note_type', 'img_urls'])
	device: str = field(default='cuda')
	torch_dtype: str = field(default='bf16')
	padding: Union[str, bool] = field(default='longest')
