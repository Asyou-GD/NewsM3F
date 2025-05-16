import torch


class WrappedModel(torch.nn.Module):
	def __init__(self, original_model):
		super(WrappedModel, self).__init__()
		self.original_model = original_model
		self.device = original_model.device

	def forward(self, input_ids, *args, **kwargs):
		labels = kwargs.pop("labels", None)
		## # using the original model's forward method
		# output = self.original_model(input_ids, *args, **kwargs)
		# pooled_logits = output.logits

		## # using the modified forward method
		transformer_outputs = self.original_model.model(input_ids, *args, **kwargs)
		hidden_states = transformer_outputs[0]
		logits = self.original_model.score(hidden_states)
		batch_size = input_ids.shape[0]

		if self.original_model.config.pad_token_id is None and batch_size != 1:
			raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
		if self.original_model.config.pad_token_id is None:
			sequence_lengths = -1
		else:
			if input_ids is not None:
				# if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
				# 获取每个样本的有效长度，不要使用argmax操作
				sequence_lengths = torch.ne(input_ids, self.original_model.config.pad_token_id).int().sum(dim=-1) - 1
			else:
				sequence_lengths = -1
		pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

		probs = torch.softmax(pooled_logits, dim=-1)
		return probs


class WrappedModelFp16(torch.nn.Module):
	def __init__(self, original_model):
		super(WrappedModelFp16, self).__init__()
		self.original_model = original_model
		self.device = original_model.device

	def get_input_embeddings(self, input_ids):
		return self.original_model.get_input_embeddings()(input_ids)

	def get_scores(self, logits, input_ids):
		if not torch.is_tensor(logits):
			logits = torch.from_numpy(logits).to(self.device)
		batch_size = logits.shape[0]
		if self.original_model.config.pad_token_id is None and batch_size != 1:
			raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
		if self.original_model.config.pad_token_id is None:
			sequence_lengths = -1
		else:
			if input_ids is not None:
				# if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
				# 获取每个样本的有效长度，不要使用argmax操作
				sequence_lengths = torch.ne(input_ids, self.original_model.config.pad_token_id).int().sum(dim=-1) - 1
			else:
				sequence_lengths = -1
		pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
		probs = torch.softmax(pooled_logits, dim=-1)
		return probs

	def forward(self, inputs_embeds, attention_mask):
		## # using the modified forward method
		transformer_outputs = self.original_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
		hidden_states = transformer_outputs[0]
		logits = self.original_model.score(hidden_states)
		return logits