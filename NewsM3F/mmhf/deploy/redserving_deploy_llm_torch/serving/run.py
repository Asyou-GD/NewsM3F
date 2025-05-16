from multiprocessing import Process
import os
import sys
import signal
import time
import socket
from subprocess import Popen, PIPE, STDOUT
import logging

from loguru import logger
import yaml
import argparse
import traceback
import re

from svc_downloader import parse_svc_and_download, download_svc

logging.basicConfig(level=logging.INFO, format='\n%(message)s')

global child_pids
global got_redserving_pid
global debug

def setenv_if_not(env_k, env_v):
    if env_k not in os.environ:
        os.environ[env_k] = env_v

def replenish_env():
    """
    补齐环境变量
    1. 配置event-client-cpp运行所需的环境变量
       参考https://code.devops.xiaohongshu.com/dataflow/events-client-cpp#%E7%89%A9%E7%90%86%E6%9C%BA%E6%88%96%E8%99%9A%E6%8B%9F%E6%9C%BA%E6%8E%A5%E5%85%A5%E6%96%B9%E5%BC%8F

    2. 配置kafka-cpp运行所需的环境变量
       参考https://events-docs.devops.xiaohongshu.com/sdk/kafka_cpp_sdk.html
    """
    # mq
    setenv_if_not("APPID"           , "redserving-test")
    setenv_if_not("XHS_ENV"         , "prod")
    setenv_if_not("XHS_REGION"      , "qc-sh")
    setenv_if_not("XHS_ZONE"        , "qcsh4")
    # kafka(developing)
    setenv_if_not("APPID"           , "redserving-test")
    setenv_if_not("XHS_SERVICE"     , "redserving-test-default")
    setenv_if_not("EDS_HOST"        , "10.4.44.53:80")
    setenv_if_not("EDS_HTTP_HOST"   , "10.0.215.163:8085")
    setenv_if_not("XHS_REGION"      , "qc-sh")
    setenv_if_not("XHS_ZONE"        , "qcsh4")
    setenv_if_not("XHS_ENV"         , "sit")
    setenv_if_not("NAMESPACE"       , "DEFAULT")

def sigint_handler(signum, frame):
    #info(f'sigint_handler: Received signal {signum} on frame {frame}')
    global debug
    if debug:
        kill_processes_immediately()
    else:
        kill_processes(15)
    sys.exit(0)

def sigtstp_handler(signum, frame):
    #info(f'sigtstp_hangler: Received signal {signum} on frame {frame}')
    # ... release your resources here ...
    # Remove our handler
    signal.signal(signum, signal.SIG_DFL)
    # Resend signal to pause process
    os.kill(os.getpid(), signum)
    # Back from suspend -- reestablish the handler.
    signal.signal(signal.SIGTSTP, sigtstp_handler)

def sigcont_handler(signum, frame):
    #info(f'sigcont_handler: Received signal {signum} on frame {frame}')
    time.sleep(0.5) # acquire resources here ...
    logger.info('Ready to go')

def get_child_pids():
    parent_id = os.getpid()
    ps_command = Popen("ps -o pid --ppid %d --noheaders" % parent_id, shell=True, stdout=PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    pid_list = []
    for pid_str in ps_output.strip().decode("utf-8").split("\n"):
        pid_list.append(int(pid_str.strip()))

    return pid_list

def kill_processes(sig=9, exit_wait=100):
    logger.info("[Notice] Start kill all related processe with sig:{}".format(sig))

    parent_id = os.getpid()
    logger.info("parent_id: {} child_pids:{}".format(parent_id, child_pids))

    if sig == 15:
        for pid in child_pids:
            logger.info("kill pid: {}".format(pid))
            os.kill(pid, signal.SIGTERM)
            break
    logger.info("sleep for {} s".format(exit_wait))
    time.sleep(exit_wait)
    logger.info("exit now")
    sys.exit(0)

def kill_processes_immediately():
    logger.info(f"[Notice] Start kill all related processes")
    cmd = "ps aux | grep 'red_serving' | grep -v 'grep' | awk '{print $2}' | xargs -i kill -9 {}"
    p = Popen(cmd, stdout=PIPE, shell=True)

    cmd = "ps aux | grep 'stub' | grep -v 'grep' | awk '{print $2}' | xargs -i kill -9 {}"
    p = Popen(cmd, stdout=PIPE, shell=True)

def exec_command(command):
    global got_redserving_pid
    process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)

    if got_redserving_pid is not True:
        global child_pids
        pids = get_child_pids()
        for p in pids:
            child_pids.append(p)
        logger.info("child_pids {}".format(child_pids))

    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            try:
                print(line.decode().strip())
            except UnicodeDecodeError:
                print(line)
    exitcode = process.wait()
    return process, exitcode

def check_port_in_use(port='8081', host='127.0.0.1', timeout=5):
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, int(port)))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()

def save_caesar_config_to_file(cmdlines, svc_args):
    SERVER_FLAG="rpc_interface_type"
    server_config=None
    for arg in cmdlines:
        if SERVER_FLAG in arg:
            server_config=arg
    
    if server_config is not None:
        try:
            if "caesar_server" in server_config:   # using caesar service
                status = True
                svc_config = download_svc(svc_args)
                if svc_config is None:
                    print("svc_config is None!, svc_args: {}".format(svc_args))
                    status=False
                else:
                    if "titan_caesar" in svc_config:
                        titan_caesar_config = svc_config["titan_caesar"]
                        if "bls_path" in titan_caesar_config:
                            bls_path = titan_caesar_config["bls_path"]
                            idx = bls_path.rfind("/")
                            bls_path = bls_path[:idx]
                            config_path = os.path.join(bls_path, "config.yaml")
                            with open(config_path, 'w') as file:
                                yaml.dump(svc_config, file)
                        else:
                            print("no bls_path config! titan_caesar_config: {}".format(titan_caesar_config))
                            status=False
                    else:
                        print("no titan_caesar config! svc_config: {}".format(svc_config))
                        status=False
                if status:
                    print("save caesar config success!")
                else:
                    print("save caesar config failed")
        except:
            traceback.print_exc()


if __name__ == '__main__':
    global child_pids
    global got_redserving_pid

    # pre-downloads the models from svc
    parser = argparse.ArgumentParser(description='bootstrapper for RedServing System')
    parser.add_argument('--llm_model', type=str, default = None)
    parser.add_argument('--quant_type', type=str, default = "origin")
    parser.add_argument('--gpus', type=int, default = 1)
    parser.add_argument('--llm_cfg', type=str, default="/workspace/configs/llm/llm.yaml")
    parser.add_argument('--svc_config_path', type=str, default = './svc.yaml')
    parser.add_argument('--sys_config_path', type=str, default = None)
    parser.add_argument('--apollo_id',type=str,default=None)
    parser.add_argument('--apollo_cluster',type=str,default=None)
    parser.add_argument('--apollo_ns',type=str,default=None)
    parser.add_argument('--apollo_env',type=str,default=None)
    parser.add_argument('--apollo_svc',type=str,default=None)
    parser.add_argument('--apollo_sys',type=str,default=None)
    parser.add_argument('--workflow_init_wait', type=int, default=None)
    parser.add_argument('--stub_init_sync_timeout', type=int, default=99999)
    parser.add_argument('--process_timeout', type=int, default=20000)
    parser.add_argument('--debug', '-d', action="store_true", default=False)
    args, unknown = parser.parse_known_args()

    child_pids = []
    got_redserving_pid = False

    ## generate cmd for boot RedServing
    logger.info(sys.argv)

    global debug
    debug = args.debug

    logger.info("args.debug: {}".format(args.debug))

    if debug:
        ## for kafka test
        replenish_env()

    if args.llm_model is not None:
        llm_cfg = None
        with open(args.llm_cfg,'r') as stream:
            llm_cfg = yaml.safe_load(stream)
        
        if args.llm_model in llm_cfg:
            if args.quant_type is not None and args.quant_type in llm_cfg[args.llm_model]:
                if args.gpus is not None and args.gpus in llm_cfg[args.llm_model][args.quant_type]:
                    args.svc_config_path = llm_cfg[args.llm_model][args.quant_type][args.gpus]

        if args.svc_config_path is None:
            logger.error("Cannot find llm svc.yaml")
            sys.exit(1)
        
        if args.workflow_init_wait is None:
            args.workflow_init_wait = 300 # s
        
        if args.stub_init_sync_timeout is None:
            args.stub_init_sync_timeout = 300 # s
        
        if args.process_timeout is None:
            args.process_timeout = 600000 # ms

    print(args)
    
    _, status = parse_svc_and_download(args.svc_config_path, args.apollo_svc, args.apollo_ns)
    if status is not True:
        logger.error("fail to download model")
        sys.exit(-1)

    ## generate cmd for boot RedServing
    rs_cmd = 'red_serving ' + ' '.join(unknown)
    stub_cmd = 'python3 /workspace/stub_boot.py '
    for arg in vars(args):
        if arg in ["debug","llm_model","llm_cfg","quant_type","gpus"]:
            continue
        if getattr(args, arg) is not None:
            rs_cmd += " --" + arg + "=" + str(getattr(args, arg))

    # for titan caesar service. DO NOT delete it!
    save_caesar_config_to_file(sys.argv[1:], args)
    
    # Assign handlers to signals
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTSTP, sigtstp_handler)
    signal.signal(signal.SIGCONT, sigcont_handler)

    p = Process(target=exec_command, args=(rs_cmd,))
    p.start()

    got_redserving_pid = True

    q = Process(target=exec_command, args=(stub_cmd,))
    q.start()

    logger.info(f"[Notice] Tap 'Ctrl+C' to exit all processes.")

    p.join()
    logger.info("[Warning] redserving exited")
    q.join()
    logger.info("[Warning] python_stub exited")
    