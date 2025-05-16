import os
# 设置PY_RED_SERVING_DEBUG环境变量
os.environ['PY_RED_SERVING_DEBUG'] = "TRUE"
import traceback
import argparse
import requests

from aimedia.modelinterface.ttypes import ResourceRequest, ResourceResponse, Feature, FeatureType
from aimedia.modelinterface.ttypes import RequestType as ThriftRequestType
from py_red_serving import PyRedServing
from py_infer_params import RequestType
from py_red_serving_utils import generate_params

def set_request(request):
    # 设置传参, 单独设置为一个函数，以便拷贝到 remote_debug.py 中
    data = {
        'img_urls': [
            "http://ci.xiaohongshu.com/1040g2sg30t4ukle73s5g5n34rifk4l7rih3de80?imageView2/2/w/1080/format/jpg",
            "http://ci.xiaohongshu.com/1040g00830t4ukjt84a7g5oi1bni41v6aempqrcg?imageView2/2/w/1080/format/jpg",
        ],
        "title"           : "宿舍4人！新品平价电动牙刷测评!（避雷版）",
        "content"         : '''#情侣电动牙刷[话题]##性价比高的电动牙刷[话题]# #电动牙刷测评[话题]# #好用电动牙刷推荐[话题]# #刷牙[话题]# #电动牙刷正确使用方法[话题]# #学生党[话题]# #平价好物[话题]# #电动牙刷欧乐b[话题]# #电动牙刷飞利浦[话题]# #电动牙刷usmile[话题]# #电动牙刷送女友[话题]# #小米电动牙刷[话题]# #飞科电动牙刷[话题]# #素士电动牙刷[话题]# #罗曼电动牙刷[话题]# #牙齿保
护计划[话题]# #口腔卫生[话题]# #牙齿问题[话题]# #有牙渍怎么办[话题]# #电动牙刷怎么用[话题]# #牙刷怎么选[话题]# #牙刷软毛好还是硬毛好[话题]# #牙刷推荐电动[话题]# #电动牙刷怎么刷[话题]# #小米电动牙刷[话题]# #平价电动牙刷[话题]#''',
	    "ocr"             : ", Usmile 高露洁, 雅澳f8, 罗曼, usmile 高露洁, usmile 高露洁, 雅澳f8, 雅澳f8, 罗曼, Usmile, 完全就是冲着颜值入的机身, 是罗马柱设计，很有质感，, 比较防滑，清洁力中规中矩, 续航蛮持久，适合_, yaao|讀, 雅澳f8, 波浪形刷毛，清洁力相当不, 错，牙缝里外都刷得很干净, 震频也比较舒适，敏感牙放, 心用，_, yaa0|ξ清, Colgate, GL/NT, 悦光, 光隆心动·悦已入, 高露洁, 渐变配色很好看，还有蓝色, 的，蛮适合当情侣款，刷毛, 磨圆率比较高，用起来很柔, 软舒服，敏感, Colgate, GLI/VT·悦光, 光ξ心动·悦入提",
        "note_type"       : "1",
        "taxonomy1"       : "3",
        "description"     : "description",
        "num_file_keys"   : "5",
        "num_file_id_list": "5"
    }
    urls = data['img_urls']

    # requestData是字典类型，并且数据项都是string
    get_imgbin = lambda url: requests.get(url.rsplit('?')[0]).content
    request.featureData = {
        'binaryDatas': Feature(type=FeatureType.BINARY_LIST, binaryListValue=[get_imgbin(u) for u in urls]),
        'imageUrls'  : Feature(type=FeatureType.STRING_LIST, stringListValue=urls),
    }
    request.requestData = {
        k: v if k != 'img_urls' else ','.join(v) for k, v in data.items()
    }
    return request

def generate_resource_request():
    request = ResourceRequest()
    request.serviceName = "SYS_TESTING"
    request.serviceType = "SYS_TESTING"
    request.requestType = ThriftRequestType.EXTRA_DATA
    return request

def run(config_path):
    try:
        # 1. 创建一个测试执行器
        # import ipdb; ipdb.set_trace()
        params = generate_params(config_path)
        py_engine = PyRedServing(params)
        # 2. 构建RPC请求对象进行模拟调用测试
        request = generate_resource_request()
        # 3. 按业务需要，往requestData填写测试数据
        request = set_request(request)
        # 4. 运行测试
        response_list = py_engine.processor(request, RequestType.RPCRESOURCE)
        return response_list
    except:
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rpc client')
    parser.add_argument('--svc_config_path', type=str, required=False, default='./svc.yaml', help='path of svc.yaml')
    args = parser.parse_args()

    response_list = run(args.svc_config_path)
    print("\n done \n")