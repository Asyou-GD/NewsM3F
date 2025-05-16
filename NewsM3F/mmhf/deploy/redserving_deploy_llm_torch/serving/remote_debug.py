# -*- coding: utf-8 -*-
import json
import argparse
import random
import string

import numpy as np
import signal
import threading
import traceback
import time
import requests

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from aimedia.innermodelinterface import InnerAIModelService
from aimedia.innermodelinterface.ttypes import InnerResourceRequest
from aimedia.modelinterface.ttypes import ResourceRequest, RequestType, ResourceResponse, Feature, FeatureType
from infra.rpc.base.ttypes import Context


def set_request(request, is_random_input=False):
	# 设置传参, 单独设置为一个函数，以便拷贝到 remote_debug.py 中
	data = {
		'img_urls': [
			"http://xhsci-10008268.cos.ap-shanghai.myqcloud.com/1040g00830tctr1pg4a6g5n5sfai5i4uveksrql0",
			# "http://ci.xiaohongshu.com/1040g00830t4ukjt84a7g5oi1bni41v6aempqrcg?imageView2/2/w/1080/format/jpg",
		],
		"title"           : "宿舍4人！新品平价电动牙刷测评!（避雷版）",
        "content"         : '''#情侣电动牙刷[话题]##性价比高的电动牙刷[话题]# #电动牙刷测评[话题]# #好用电动牙刷推荐[话题]# #刷牙[话题]# #电动牙刷正确使用方法[话题]# #学生党[话题]# #平价好物[话题]# #电动牙刷欧乐b[话题]# #电动牙刷飞利浦[话题]# #电动牙刷usmile[话题]# #电动牙刷送女友[话题]# #小米电动牙刷[话题]# #飞科电动牙刷[话题]# #素士电动牙刷[话题]# #罗曼电动牙刷[话题]# #牙齿保
护计划[话题]# #口腔卫生[话题]# #牙齿问题[话题]# #有牙渍怎么办[话题]# #电动牙刷怎么用[话题]# #牙刷怎么选[话题]# #牙刷软毛好还是硬毛好[话题]# #牙刷推荐电动[话题]# #电动牙刷怎么刷[话题]# #小米电动牙刷[话题]# #平价电动牙刷[话题]#''',
	    "ocr"             : ", Usmile 高露洁, 雅澳f8, 罗曼, usmile 高露洁, usmile 高露洁, 雅澳f8, 雅澳f8, 罗曼, Usmile, 完全就是冲着颜值入的机身, 是罗马柱设计，很有质感，, 比较防滑，清洁力中规中矩, 续航蛮持久，适合_, yaao|讀, 雅澳f8, 波浪形刷毛，清洁力相当不, 错，牙缝里外都刷得很干净, 震频也比较舒适，敏感牙放, 心用，_, yaa0|ξ清, Colgate, GL/NT, 悦光, 光隆心动·悦已入, 高露洁, 渐变配色很好看，还有蓝色, 的，蛮适合当情侣款，刷毛, 磨圆率比较高，用起来很柔, 软舒服，敏感, Colgate, GLI/VT·悦光, 光ξ心动·悦入提",
        "note_type"       : "1",
		"taxonomy1": "科技数码",
		"description": "科技数码[SEP]自信至上禁止低头[SEP]",
		"num_file_keys": "0",
		"num_file_id_list": "1"
	}

	if is_random_input:
		tmp_inputs = {
			"title": ' '.join(random.choice(string.ascii_lowercase) for _ in range(20)),
			"content": ' '.join(random.choice(string.ascii_lowercase) for _ in range(200)),
			"ocr": ' '.join(random.choice(string.ascii_lowercase) for _ in range(4000)),
		}
		data.update(tmp_inputs)

	urls = data['img_urls']

	# requestData是字典类型，并且数据项都是string
	get_imgbin = lambda url: requests.get(url.rsplit('?')[0]).content
	request.featureData = {
		'binaryDatas': Feature(type=FeatureType.BINARY_LIST, binaryListValue=[get_imgbin(u) for u in urls]),
		'imageUrls': Feature(type=FeatureType.STRING_LIST, stringListValue=urls),
	}
	request.requestData = {
		k: v if k != 'img_urls' else ','.join(v) for k, v in data.items()
	}
	return request


## global
stop = False

succ_cost = []
fail_cost = []

lock = threading.Lock()


def signal_handler(signum, frame):
	print(f'signal_handler: Received signal {signum} on frame {frame}')
	global stop
	stop = True


def count_and_print_stat():
	global succ_cost
	global fail_cost
	global stop
	global lock

	while True:
		time.sleep(1)
		if stop:
			return
		try:
			lock.acquire()
			succ_cost.sort()
			fail_cost.sort()

			succ_cnt = len(succ_cost)
			fail_cnt = len(fail_cost)

			total_succ_cost = 0.0
			for cost in succ_cost:
				total_succ_cost += cost

			total_fail_cost = 0.0
			for cost in fail_cost:
				total_fail_cost += cost

			succ_p99_idx = int(succ_cnt * 0.99)
			succ_p99_idx = succ_p99_idx - 1 if succ_p99_idx < succ_cnt else succ_cnt - 1
			succ_p99_idx = succ_p99_idx if succ_p99_idx >= 0 else 0

			fail_p99_idx = int(fail_cnt * 0.99)
			fail_p99_idx = fail_p99_idx - 1 if fail_p99_idx < fail_cnt else fail_cnt - 1
			fail_p99_idx = fail_p99_idx if fail_p99_idx >= 0 else 0

			succ_cost = [0.0] if succ_cnt <= 0 else succ_cost
			fail_cost = [0.0] if fail_cnt <= 0 else fail_cost

			succ_cost_avg = 0.0 if succ_cnt <= 0 else round(total_succ_cost / succ_cnt, 3)
			succ_cost_p99 = succ_cost[succ_p99_idx]

			fail_cost_avg = 0.0 if fail_cnt <= 0 else round(total_fail_cost / fail_cnt, 3)
			fail_cost_p99 = fail_cost[fail_p99_idx]

			print("*" * 60)
			print(
				"\nsucc_qps: {}\t succ_cost_avg: {}(ms)\t succ_cost_p99: {}(ms)\nfail_qps: {}\t fail_cost_avg: {}(ms)\t fail_cost_p99: {}(ms)\n".format(
					succ_cnt, succ_cost_avg, succ_cost_p99, fail_cnt, fail_cost_avg, fail_cost_p99))
			print("*" * 60)
			print("\n")
		except:
			traceback.print_exc()
		finally:
			succ_cost = []
			fail_cost = []
			lock.release()


def add_stat(cost, succ=True):
	global succ_cost
	global fail_cost
	global lock
	try:
		lock.acquire()
		if succ:
			succ_cost.append(round(cost * 1000.0, 3))
		else:
			fail_cost.append(round(cost * 1000.0, 3))
	finally:
		lock.release()


def createThriftClient(ip='127.0.0.1', port=8080, timeout=500):
	transport = TSocket.TSocket(ip, port)
	transport.setTimeout(timeout)
	transport = TTransport.TFramedTransport(transport)
	protocol = TBinaryProtocol.TBinaryProtocol(transport)
	client = InnerAIModelService.Client(protocol)
	return transport, client


def createResourceRequest(service_type="SYS_TESTING", service_name="SYS_TESTING", req_type=RequestType.EXTRA_DATA):
	req = ResourceRequest()
	req.serviceType = service_type  # 按服务实际需要设定，非安审团队服务可随意设置
	req.serviceName = service_name  # 按服务实际需要设定，非安审团队服务可随意设置
	req.requestType = req_type
	return req


def request(loop_count, ip, port, timeout, verbose):
	global stop
	try:
		transport, client = createThriftClient(ip, port, timeout)
		transport.open()
	except:
		print(traceback.format_exc())
		return

	ctx = Context()
	ctx.traceID = "3333"  # 测试时可随意指定
	resource_req = createResourceRequest()

	while True:
		succ = True
		cost = 0.0
		try:
			for i in range(loop_count):
				## example
				resource_req = set_request(resource_req)

				real_req = InnerResourceRequest()
				real_req.context = ctx
				real_req.resourceRequest = resource_req

				st = time.time()
				ret = client.resourceQuery(real_req)
				cost = time.time() - st
				if verbose:
					print(ret)
				add_stat(cost, succ)
				cost = 0.0
		except:
			traceback.print_exc()
			succ = False
			add_stat(cost, succ)
		if stop:
			return


def main():
	global stop

	parser = argparse.ArgumentParser(description='rpc client')
	parser.add_argument('-ip', '--ip', type=str, required=False, default='127.0.0.1',
	                    help='IP of rpc-server, default=127.0.0.1')
	parser.add_argument('-p', '--port', type=int, required=False, default=8081,
	                    help='Port of  rpc-server, default=8081')
	parser.add_argument('-thds', '--threads', type=int, required=False, default=0, help='Threads of client, default=0')
	parser.add_argument('-lc', '--loop_count', type=int, required=False, default=1, help='Number of loops.')
	parser.add_argument('-t', '--timeout', type=int, required=False, default=500, help='RPC timeout(ms).')
	parser.add_argument('-v', '--verbose', type=bool, required=False, default=False, help='Print rpc response.')
	args = parser.parse_args()

	# Assign handlers to signals
	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTSTP, signal_handler)
	signal.signal(signal.SIGCONT, signal_handler)

	if args.threads == 0:
		stop = True
		request(args.loop_count, args.ip, args.port, args.timeout, args.verbose)
	else:
		thds = []
		stat_thd = threading.Thread(target=count_and_print_stat, args=())
		stat_thd.start()

		for i in range(args.threads):
			thd = threading.Thread(target=request,
			                       args=(args.loop_count, args.ip, args.port, args.timeout, args.verbose))
			thd.start()
			thds.append(thd)
		for i in range(args.threads):
			thds[i].join()
		stat_thd.join()


if __name__ == '__main__':
	main()
