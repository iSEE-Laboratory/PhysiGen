import sys
sys.path.append(sys.path[0]+r"/../")
import numpy as np
import torch

from datetime import datetime
from datasets import get_dataset_motion_loader, get_motion_loader
from models import *
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils.utils import *
from configs import get_config
from os.path import join as pjoin
from tqdm import tqdm

import argparse


os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import scipy.ndimage.filters as filters
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collision.coll_metric_mesh import coll_metric_mesh
from smplx import SMPLXLayer, SMPLHLayer
from visualize.utils.mapping import (INTERX_TO_SMPLX, INTERX_TO_SMPLH, JointMapper)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 导入 peft 库的相关模块
from peft import LoraConfig, get_peft_model, TaskType

warnings.filterwarnings("ignore", message="You are using a SMPL+H model")
body_models = SMPLHLayer(model_path='./smpl_models/smplh',num_betas=10,gender='neutral',
                        joint_mapper=JointMapper(INTERX_TO_SMPLH),flat_hand_mean=False,ext='npz')


num_gpus = 3 
gpu_id_list = [1,5,6]  
max_workers = 8 

def _worker(args):
    """
    args = (global_idx, motion1_np, motion2_np, motion_lens, gpu_id)
    """
    idx, m1_np, m2_np, mlen, gpu_id, name,text,replication,vi_video = args

    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{int(gpu_id)}')
    coll_dis, coll_frame, coll_dis_wohand, coll_frame_wohand = coll_metric_mesh(body_models, m1_np, m2_np, mlen, device, name,text,replication,vi_video)

    return idx, float(coll_dis), float(coll_frame), float(coll_dis_wohand), float(coll_frame_wohand)

def parallel_collisions(motion_loader, num_gpus=8, replication=0, vi_video = False,mode = 'pre'):
    tasks = []
    global_idx = 0
    num_dataset = 0
    for batch in tqdm( motion_loader):
        name = batch[0]
        text = batch[1]
        B = batch[2].shape[0]
        num_dataset += B
        for b in range(B):
            mlen = batch[4][b]
            # 1.1 拆 numpy + 滤波
            m1 = batch[2][b,:int(mlen),:22*3].reshape(-1,22,3).numpy()
            m2 = batch[3][b,:int(mlen),:22*3].reshape(-1,22,3).numpy()
            # if mode == 'pre':
            #     m1 = filters.gaussian_filter1d(m1, 1, axis=0, mode='nearest')
            #     m2 = filters.gaussian_filter1d(m2, 1, axis=0, mode='nearest')


            gpu_id_index = global_idx % num_gpus 
            gpu_id = gpu_id_list[gpu_id_index]
            
            tasks.append((global_idx, m1, m2, int(mlen), gpu_id, name[b], text[b], replication,vi_video))
            global_idx += 1


    coll_dis_all = 0
    coll_frame_all = 0
    coll_dis_all_wohand = 0
    coll_frame_all_wohand = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as exe:
        futures = [exe.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures)):
            idx, cd, cf, cd_wohand, cf_wohand = fut.result()
            coll_dis_all   += cd
            coll_frame_all += cf
            coll_dis_all_wohand += cd_wohand
            coll_frame_all_wohand += cf_wohand
            print('finished one')

    return coll_dis_all, coll_frame_all, coll_dis_all_wohand, coll_frame_all_wohand,  num_dataset



def evaluate_collision(motion_loaders, file,device, replication,vi_video):
    eval_dict = OrderedDict({})
    print('========== Evaluating collision ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        coll_dis_all = 0
        coll_frame_all = 0
        coll_dis_all_wohand = 0
        coll_frame_all_wohand = 0
        with torch.no_grad():
            num_dataset = 0
            if motion_loader_name == 'TIMotion':
                coll_dis_all, coll_frame_all, coll_dis_all_wohand, coll_frame_all_wohand, num_dataset = parallel_collisions(motion_loader, num_gpus=num_gpus, replication=replication,vi_video=vi_video)
                print('finished TIMotion')
            elif 'truth' in motion_loader_name:
                coll_dis_all, coll_frame_all, coll_dis_all_wohand, coll_frame_all_wohand, num_dataset = parallel_collisions(motion_loader, num_gpus=num_gpus, mode = 'gt')
                print('finished gt')
                    
        print("num_dataset",num_dataset)
        coll_dis_all = coll_dis_all / num_dataset
        coll_frame_all = coll_frame_all / num_dataset
        coll_dis_all_wohand = coll_dis_all_wohand / num_dataset
        coll_frame_all_wohand = coll_frame_all_wohand / num_dataset

        print(f'---> [{motion_loader_name}] coll_dis_all: {coll_dis_all:.4f}')
        print(f'---> [{motion_loader_name}] coll_dis_all: {coll_dis_all:.4f}', file=file, flush=True)
        print(f'---> [{motion_loader_name}] coll_frame: {coll_frame_all:.4f}')
        print(f'---> [{motion_loader_name}] coll_frame: {coll_frame_all:.4f}', file=file, flush=True)
        print(f'---> [{motion_loader_name}] coll_dis_all_wohand: {coll_dis_all_wohand:.4f}')
        print(f'---> [{motion_loader_name}] coll_dis_all_wohand: {coll_dis_all_wohand:.4f}', file=file, flush=True)
        print(f'---> [{motion_loader_name}] coll_frame_all_wohand: {coll_frame_all_wohand:.4f}')
        print(f'---> [{motion_loader_name}] coll_frame_all_wohand: {coll_frame_all_wohand:.4f}', file=file, flush=True)


        eval_dict[motion_loader_name+' coll_dis_all'] = coll_dis_all
        eval_dict[motion_loader_name+' coll_frame_all'] = coll_frame_all
        eval_dict[motion_loader_name+' coll_dis_all_wohand'] = coll_dis_all_wohand
        eval_dict[motion_loader_name+' coll_frame_all_wohand'] = coll_frame_all_wohand
    return eval_dict

