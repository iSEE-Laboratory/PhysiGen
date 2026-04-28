import torch
import numpy as np
from ipdb import set_trace

def init_params(params,body_models,nf,device):
    """
    Using pose mean to initialize params,and move to cuda
    """
    for key in params.keys():
        params[key]=torch.tensor(params[key],dtype=torch.float32,
                                 device=device)
    params['left_hand_pose']=body_models.left_hand_mean\
                            .reshape((15,3))[None,...].expand([nf,-1,-1]).clone().to(device)
    params['right_hand_pose']=body_models.right_hand_mean\
                            .reshape((15,3))[None,...].expand([nf,-1,-1]).clone().to(device)
    return params

def tensor_to_numpy(params):
    for key in params.keys():
        params[key]=params[key].detach().cpu().numpy()
    return params


def tensor_to_array(params):
    save_keys = ['global_orient', 'body_pose', 'lhand_pose', 'rhand_pose', 'transl']
    res = []
    for key in save_keys:
        tmp = params[key].detach().cpu().numpy()
        if len(tmp.shape) == 2:
            tmp = tmp[:,None]
        res.append(tmp)
    res = np.concatenate(res, axis=1)
    return res


def numpy_to_tensor(params,device):
    for key in params.keys():
        params[key]=torch.tensor(params[key],dtype=torch.float32,
                                 device=torch.device(device))
    return params
