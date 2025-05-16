# Hugging Face Trainer

## 数据集制作
1. 解析parquet格式为arrow格式并保存到磁盘

注：arrow文件有很多分片，每一行是一个样本，每一列是一个特征，特征名为列名，特征值为列值。

```shell
python mmhf/tools/datasets/trainval_data_2_disk.py
```
2. 准备文本或者多模态模型的训练集

注：
- 训练集采用Webdataset的Tar包组织形式，每个Tar包包含多个样本，每个样本是一个字典，字典的key是特征名，value是经过编码的特征值。
- 测试集仍然采用arrow格式，每一行是一个样本，每一列是一个特征，特征名为列名，特征值为列值。

```shell
python mmhf/tools/datasets/webdataset/convert_2_webdataset.py
```

注：
- 生成的训练集和测试集按日期分为多个Tar包，每次运行都会在此基础上增加新日期的Tar包。

3. 计算样本分布

```shell
python mmhf/tools/datasets/cal_data_distribution.py
```

## 训练过程

1. 依赖包

```shell
pip install -U transformers peft trl datasets rich accelerate deepspeed

# 可选 pip install flash-attn --no-build-isolation  Torch2.X已经集成了flash-attn
```

2. 预训练模型下载
    
```shell
pip install modelscope
# modelscope login  --token e948630d-db13-4917-99e0-3d5122538aed
modelscope download --model 'OpenGVLab/InternVL2-1B'  --local_dir work_dirs/model_cache/InternVL2-1B
```

3. 配置文件位置

配置文件位于`mmhf/configs`目录下，配置文件的格式为Py格式。

4. 训练脚本


```shell
# 单机
accelerate launch  --config_file mmhf/configs/gpu_8_deepspeed.yaml mmhf/train_by_trainer_llm.py

accelerate launch  --config_file mmhf/configs/gpu_8_deepspeed.yaml mmhf/train_by_trainer_vlm.py

# 多机
accelerate launch  --config_file mmhf/configs/gpu_16_deepspeed_rank0.yaml mmhf/train_by_trainer_llm.py
accelerate launch  --config_file mmhf/configs/gpu_16_deepspeed_rank1.yaml mmhf/train_by_trainer_llm.py

accelerate launch  --config_file mmhf/configs/gpu_16_deepspeed_rank0.yaml mmhf/train_by_trainer_vlm.py
accelerate launch  --config_file mmhf/configs/gpu_16_deepspeed_rank1.yaml mmhf/train_by_trainer_vlm.py
```




## 测试过程

1. 测试脚本

```shell
# 单机
accelerate launch  --config_file mmhf/configs/gpu_8_deepspeed.yaml mmhf/test_llm.py
# 或者
python mmhf/test_llm.py
```


## 部署

注：直接用bf16模型进行推理即可

1. 对齐模型

```shell
python mmhf/deploy/convert_model/hf_merge_weights_and_upload.py
```

2. 部署采用的镜像

```shell
docker-reg.devops.xiaohongshu.com/media/cky:trt861_1.14.5_vlm_v1
```

3. LLM部署代码

```shell
https://code.devops.xiaohongshu.com/content-quality/market_deploy/-/tree/cky_deploy_llm_torch
```


