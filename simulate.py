import requests
import threading
import torch
import pickle
from arguments import parser
from models import Model
from utils import create_buffers, act, get_batch
from torch import multiprocessing as mp
import zlib
import time


class Simulate(object):

    def __init__(self, flags):
        self.flags = flags
        self.url_v = "http://zx081325.xyz:81/zx/model_version.txt"
        self.url_w = "http://zx081325.xyz:81/zx/white_weights.ckpt"
        self.url_b = "http://zx081325.xyz:81/zx/black_weights.ckpt"
        self.model_path_w = 'model/white_weights.ckpt'
        self.model_path_b = 'model/black_weights.ckpt'
        self.model = Model(device='cpu', board_size=flags.board_size)
        self.version = 0

    # 从云盘下载模型
    def download_pkl(self):
        try:
            res = requests.get(self.url_v).content
            if int(res) <= self.version:
                return
            print("version:", int(res))
            self.version = int(res)
            start_time = time.time()
            res = requests.get(self.url_w)
            with open(self.model_path_w, "wb") as f:
                f.write(res.content)
            print("white模型下载成功, 下载用时:{}".format(time.time() - start_time))
            start_time = time.time()
            res = requests.get(self.url_b)
            with open(self.model_path_b, "wb") as f:
                f.write(res.content)
            print("black模型下载成功, 下载用时:{}".format(time.time() - start_time))
            self.load_model()
        except Exception as e:
            print("下载模型失败:", repr(e))

    # 加载模型
    def load_model(self):
        from models import get_model_dict
        for position in ['white', 'black']:
            model_dict = get_model_dict(self.flags.board_size)
            model = model_dict[position]()
            if position == "white":
                model_state_dict = torch.load(self.model_path_w, map_location='cpu')
            else:
                model_state_dict = torch.load(self.model_path_b, map_location='cpu')
            model.load_state_dict(model_state_dict)
            model.eval()
            self.model.models[position] = model
            print("加载" + position + "模型成功")

    # 生成对局数据
    def simulate(self):
        if not self.flags.actor_device_cpu:
            if not torch.cuda.is_available():
                raise AssertionError(
                    "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")

        if self.flags.actor_device_cpu:
            device_iterator = ['cpu']
        else:
            device_iterator = range(self.flags.num_actor_devices)
            assert self.flags.num_actor_devices <= len(self.flags.gpu_devices.split(
                ',')), 'The number of actor devices can not exceed the number of available devices'

        # 初始化buffers
        buffers = create_buffers(self.flags, device_iterator)

        # Initialize queues
        actor_processes = []
        ctx = mp.get_context('spawn')
        free_queue = {}
        full_queue = {}

        for device in device_iterator:
            _free_queue = {'white': ctx.SimpleQueue(), 'black': ctx.SimpleQueue()}
            _full_queue = {'white': ctx.SimpleQueue(), 'black': ctx.SimpleQueue()}
            free_queue[device] = _free_queue
            full_queue[device] = _full_queue

        for device in device_iterator:
            for m in range(self.flags.num_buffers):
                free_queue[device]['white'].put(m)
                free_queue[device]['black'].put(m)

        # Starting actor processes
        for device in device_iterator:
            for i in range(self.flags.num_actors):
                actor = ctx.Process(
                    target=act,
                    args=(
                    i, device, free_queue[device], full_queue[device], self.model, buffers[device], self.flags))
                actor.start()
                actor_processes.append(actor)

        def batch_and_send(i, device, position, local_lock, lock=threading.Lock()):
            """Thread target for the learning process."""
            start_time = time.time()
            while True:
                batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position],
                                  self.flags, local_lock)
                with lock:
                    # 先序列化，再压缩
                    batch = zlib.compress(pickle.dumps(batch), level=9)
                    # 发送batch
                    data = {"file": batch}
                    requests.post(url='http://zx081325.xyz:83/{}_batch'.format(position), files=data)
                    end_time = time.time()
                    print(position, "batch发送成功", "速度: {} frames/s".format(3200 / (end_time - start_time)))
                    start_time = time.time()
                    # 更新模型
                    self.download_pkl()

        threads = []
        locks = {}
        for device in device_iterator:
            locks[device] = {'white': threading.Lock(), 'black': threading.Lock()}

        for device in device_iterator:
            for i in range(self.flags.num_threads):
                for position in ['white', 'black']:
                    thread = threading.Thread(
                        target=batch_and_send, name='batch-%d' % i,
                        args=(i, device, position, locks[device][position]))
                    thread.start()
                    threads.append(thread)


if __name__ == "__main__":
    flags = parser.parse_args(['--actor_device_cpu'])
    simulate = Simulate(flags)
    simulate.download_pkl()
    simulate.simulate()




