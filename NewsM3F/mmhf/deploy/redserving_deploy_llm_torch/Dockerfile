#FROM docker-reg.devops.xiaohongshu.com/media/mm_maket_cky:trt8.6.1.6
FROM docker-reg.devops.xiaohongshu.com/media/cky:trt861_1.14.5_vlm_v1

RUN mkdir /workspace/redserving_py
COPY ./serving /workspace/redserving_py
COPY ./requirements.txt /workspace/redserving_py/


RUN pip config set global.index-url http://pypi.xiaohongshu.com/simple/ \
    && pip config set global.trusted-host pypi.xiaohongshu.com \
    && pip install -r /workspace/redserving_py/requirements.txt

ENV PYTHONPATH /workspace/redserving_py:$PYTHONPATH
#ENV LD_LIBRARY_PATH /workspace/TensorRT86/lib:$LD_LIBRARY_PATH
#ENV PATH /workspace/TensorRT86/bin:$PATH
WORKDIR /workspace/redserving_py
RUN cd /workspace/redserving_py
