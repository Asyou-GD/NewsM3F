# 下载并运行镜像
```
docker run -d -it --gpus all --name=ones --shm-size="120g" -v /mnt:/mnt docker-reg.devops.xiaohongshu.com/media/red_serving:trt861_1.14.5 bash
docker exec -it ones bash

conda clean --tarballs
conda clean --all
pip cache purge
apt-get clean
apt-get autoremove
rm -rf /tmp/*
rm -rf ~/.cache/*

docker commit ones docker-reg.devops.xiaohongshu.com/media/cky:trt861_1.14.5_vlm_v1
docker push docker-reg.devops.xiaohongshu.com/media/cky:trt861_1.14.5_vlm_v1
```

# 上传和下载模型
```
python3 -m redserving_toolkit.transfer -t content_quality -s contentproj-noteorderquality-test -m classifier -f ./work_dirs/quality/Qwen2-1.5B_title_2024-08-16_checkpoint-2614_torch_deploy
python3 -m redserving_toolkit.transfer -d -t content_quality -s contentproj-noteorderquality-test -m classifier -f ./model
```

# local debug
```shell
python3 local_debug.py
```

# remote debug
1. 启动服务

```shell
python3 run.py --debug --logtostderr=1
```

2. 远程调试

```shell
python3 remote_debug.py --verbose True --threads 8
```



# 设置镜像源
```
pip config set global.index-url http://pypi.xiaohongshu.com/simple/
pip config set global.trusted-host pypi.xiaohongshu.com

pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
```
# 需要安装的Python包
```shell
pip3 install torch==2.4.0 torchvision torchaudio
pip install -U openmim transformers accelerate bitsandbytes 
pip install -U onnx onnxruntime-gpu peft datasets
mim install mmcv
```

# 注意事项

process_timeout=30000
workflow_init_wait=99999
stub_init_sync_timeout=99999

# 文本4096 thread2 qps 7