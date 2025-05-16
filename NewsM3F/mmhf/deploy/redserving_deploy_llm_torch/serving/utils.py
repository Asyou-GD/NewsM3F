from param_config import chat_templates


def apply_chat_template_for_seq_classification(examples, task_name='标题党'):
	title = examples["title"]
	content = examples["content"]
	ocr = examples["ocr"]
	messages = [
		{
			'role': "system",
			'content': "你是一个很有用的助手。",
		},
		{
			'role': "user",
			'content': chat_templates[task_name].format(title=title, content=content, ocr=ocr),
		},
		{
			'role': "assistant",
			'content': "",
		}
	]
	return_dict = dict(
		messages=messages,
	)
	return return_dict


def format_chat_template(examples, messages_field='messages', tokenizer=None):
	if isinstance(examples[messages_field][0], list):
		output_texts = []
		for i in range(len(examples[messages_field])):
			output_texts.append(tokenizer.apply_chat_template(examples[messages_field][i], tokenize=False))
		return output_texts
	else:
		return tokenizer.apply_chat_template(examples[messages_field], tokenize=False)


# tokenize the dataset for seq_classification
def seq_tokenizing_for_seq_classification(
		element,
		tokenizer=None,
		messages_field='messages',
		formatting_func=format_chat_template,
		add_special_tokens=False,
		max_seq_length=4096,
		padding=True,
		return_tensors='pt'
):
	outputs = tokenizer(
		element[messages_field] if formatting_func is None else formatting_func(element, messages_field, tokenizer),
		add_special_tokens=add_special_tokens,
		truncation=True,
		padding=padding,
		max_length=max_seq_length,
		return_overflowing_tokens=False,
		return_length=False,
		return_tensors=return_tensors
	)
	return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

