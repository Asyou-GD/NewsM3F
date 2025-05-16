import os
import base64
import json
import traceback
import time
import sys
from typing import List
from loguru import logger
import argparse

import oss2
import yaml

from tqdm import tqdm
from redserving_toolkit.oss_util import OssClient
from redserving_toolkit.cos_util import CosClient
from redserving_toolkit.apollo_client import InnerApolloClient

def get_version(path):
    pos = path.find('/')
    if pos == -1:
        return ""
    else:
        return path[:pos]

def drop_version(path):
    pos = path.find('/')
    if pos == -1:
        return path
    else:
        return path[pos+1:]

def check_success_file(directory):
    success_file_path = os.path.join(directory, "success")
    if os.path.exists(success_file_path):
        return True
    return None


class DownloadItem:
    def __init__(self, name: str, model_placement: str, model_repo_path: str,
                 download_type=None, region="shanghai", prefix=None, version=None,
                 model_repo_bucket=None, model_repo_secret_key=None, model_repo_access_key=None, **kwargs) -> None:
        self.name: str = name
        self.download_type: str = download_type
        self.model_placement: str = model_placement
        self.model_repo_path: str = model_repo_path
        self.region: str = region
        self.prefix: str = prefix
        self.version: str = version
        self.kwargs = kwargs

    def is_local(self):
        return self.model_placement == 'local'
    
    def get_download_type(self):
        return self.download_type
    
    def skip(self):
        return self.download_type is None
    
    def get_placement(self):
        return self.model_placement
    
    def get_remote_prefix(self):
        return os.path.join(self.model_repo_path, self.name)

    def get_local_prefix(self, default_prefix=None):
        if self.prefix:
            return self.prefix
        
        if self.is_local():
            local_model_path = os.path.join(self.model_repo_path, self.name)
            if self.version is None:
                dir_list = os.listdir(local_model_path)
                list.sort(dir_list)
                local_model_path = os.path.join(local_model_path, dir_list[-1])
            else:
                local_model_path = os.path.join(local_model_path, self.version)
            return local_model_path
        else:
            return os.path.join(default_prefix, self.name)
        
    
    def get_version(self):
        return self.version


def download_svc(args):
    svc_config = None
    try:
        if args.svc_config_path:
            with open(args.svc_config_path) as f:
                svc_config = yaml.safe_load(f)
        elif args.apollo_svc:
            client = InnerApolloClient().get_apollo_client()
            cfg_content = client.get_value(args.apollo_svc, namespace=args.apollo_ns)
            svc_config = yaml.safe_load(cfg_content)
    except:
        traceback.print_exc()
    return svc_config

def parse_svc_and_download(svc_config_path, apollo_svc=None, apollo_ns=None):
    try:
        if svc_config_path:
            with open(svc_config_path) as f:
                svc_config = yaml.safe_load(f)

        elif apollo_svc:
            client = InnerApolloClient().get_apollo_client()
            cfg_content = client.get_value(apollo_svc, namespace=apollo_ns)
            svc_config = yaml.safe_load(cfg_content)

        if svc_config is None:
            return None, False

        downloads: List[dict] = None
        download_from_settings = False
        if 'model_downloads' in svc_config:
            downloads = svc_config['model_downloads']
            logger.info("download from 'model_downloads'...")
            download_from_settings = False
        elif 'model_settings' or 'pseudo_model_settings' in svc_config:
            downloads = []
            model_settings_name = 'model_settings' if 'model_settings' in svc_config else 'pseudo_model_settings'
            for item in svc_config[model_settings_name]:
                if "rpc_config" not in item:
                    downloads.append(item)
            logger.info("download from 'model_settings'...")
            download_from_settings = True
        
        
        if downloads is None or downloads == []:
            logger.warning("downloads is None, no need to download models")
            return svc_config, True

        default_prefix = os.environ.get("SVC_DOWNLOAD_PREFIX", "/workspace/models/")
        model_deploys = {}
        for item in downloads:
            model = DownloadItem(**item)

            if not model.is_local():
                storage_client = None
                if model.get_placement() == 'oss':
                    lst = model.model_repo_path.split('/')
                    if len(lst) < 2:
                        logger.error("invalid model_repo_path: {}".format(model.model_repo_path))
                    team = lst[1]
                    storage_client = OssClient(region=model.region, team=team)
                elif model.get_placement() == 'cos':
                    storage_client = CosClient()
                
                local_prefix = model.get_local_prefix(default_prefix)
                remote_prefix = model.get_remote_prefix()

                start_time = time.time()

                with_version = False
                if os.environ.get("SVC_DOWNLOAD_WITH_VERSION"):
                    with_version = bool(int(os.environ.get("SVC_DOWNLOAD_WITH_VERSION")))
                ret = False
                try:
                    ret, download_version = storage_client.download_path(remote_prefix, local_prefix, version=model.get_version(), num_threads=10, include_version=with_version)
                except Exception as e:
                    _, team_str, service_str = model.model_repo_path.split('/')
                    if model.get_placement() == 'cos':
                        ret, download_version = storage_client.download_model(team_str, service_str, model.name, local_prefix, version=model.get_version(), num_threads=10, include_version=with_version)
                        # save to /workspace/models/cl/red-serving/content_quality/contentproj-noteorderquality-test/classifier/1723898105829/
                if ret is not True:
                    logger.error("model file download Failed! info:\n{}".format(ret))
                    return None, False

                if with_version:
                    local_prefix = os.path.join(local_prefix, download_version)
                
                logger.info("model was downloaded to {}, version is :{}, cost: {} min".format(local_prefix, download_version, (time.time() - start_time) // 60))

                model_deploys[model.name] = {
                    "path": local_prefix, "version": download_version, 
                    "remote": remote_prefix, "placement": model.get_placement(), 
                    "kwargs": model.kwargs
                }
                
                python_path = os.environ.get('PYTHONPATH', '')

                # Prepend the model path to PYTHONPATH
                if python_path:
                    python_path = local_prefix + os.pathsep + python_path
                else:
                    python_path = local_prefix
                # Set the modified PYTHONPATH
                os.environ['PYTHONPATH'] = python_path
                # Set the modified PYTHONPATH in the system environment
                python_path = os.environ.get('PYTHONPATH', '')
            else:
                model_deploys[model.name] = {
                    "path": model.get_local_prefix(), "placement": model.get_placement(), 
                    "kwargs": model.kwargs
                }

        if len(model_deploys) > 0:
            os.environ['MODEL_DOWNLOADS'] = json.dumps(model_deploys)
            logger.info("MODEL_DOWNLOADS: {}".format(os.environ['MODEL_DOWNLOADS']))

        logger.info("All Done!")
        return svc_config, True

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)
        
    return None, False


if __name__ == "__main__":
    # pre-downloads the models from svc
    parser = argparse.ArgumentParser(description='bootstrapper for RedServing System')
    parser.add_argument('--svc_config_path', type=str, default = './svc.yaml')
    parser.add_argument('--sys_config_path', type=str, default = None)
    parser.add_argument('--apollo_id',type=str,default=None)
    parser.add_argument('--apollo_cluster',type=str,default=None)
    parser.add_argument('--apollo_ns',type=str,default=None)
    parser.add_argument('--apollo_env',type=str,default=None)
    parser.add_argument('--apollo_svc',type=str,default=None)
    parser.add_argument('--apollo_sys',type=str,default=None)
    parser.add_argument('--model_preload',type=bool,default=False)

    svc_args, unknown = parser.parse_known_args()
    if parse_svc_and_download(svc_args.svc_config_path, svc_args.apollo_svc, svc_args.apollo_ns):
        sys.exit(0)
    sys.exit(-1)
