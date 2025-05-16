#!/bin/bash

##############using example##############
# bash run.sh --wait_time=120 --logtostderr=1  --process_timeout=1500 --apollo_id=redserving --apollo_ns=media.auditonline-cbtestv1-redserving --apollo_env=prod --apollo_cluster=default --apollo_svc=svc.yaml --apollo_sys=sys.yaml
# ATTENTION params format must like this: bash run.sh --xx=yy --xx=yy
#########################################

# # collect "/tmp/red_serving.INFO  /tmp/red_serving.WARNING /tmp/red_serving.ERROR" and upload to es
#fluent-bit -c /usr/local/etc/fluent-bit/fluent-bit.conf -R /usr/local/etc/fluent-bit/parsers.conf

conda activate base
source activate base
cd /workspace/redserving_py
conda env list
exec python3 -u run.py $@

