import torch
"""
This script consists of two parts:
1. Load the precomputed 3D positions from .npy file
    Notes: .npy has 64 joints of Optitrack and is downsampled to 30 fps.
2. do the SMPLify(-X) optimization and get SMPL(-X) parameters
"""

import argparse
import sys
import time

sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import torch
from smplx import SMPLXLayer, SMPLHLayer

from in2in.visualize.smplifyx.optimize import *
from in2in.visualize.utils.io import write_smplx
from in2in.visualize.utils.mapping import (INTERX_TO_SMPLX, INTERX_TO_SMPLH, JointMapper)

from in2in.visualize.utils.torch_utils import *
from ipdb import set_trace
from in2in.visualize.utils.geometry import batch_rodrigues
import trimesh
import torch
from tqdm import tqdm
from in2in.utils.rotation_conversions import rotation_6d_to_matrix

def ro6d_smplifyx(motion1, motion2, save_path, mode, device):#ori_motion1, ori_motion2, 
    '''
    motion:B,T,268
    '''
    B,T = motion1.shape[:2]
    motions = torch.cat((motion1, motion2), dim=0)
    N = B*T*2
    motions = motions.reshape((N, -1))


    if mode == 'smplh':
        nj = 22
        body_models = SMPLHLayer(model_path='./smpl_models/smplh',num_betas=10,gender='neutral',
                        joint_mapper=JointMapper(INTERX_TO_SMPLH),flat_hand_mean=False,ext='npz').to(device)

        motions_transl = motions[:,:3]# [B,T,3]
        motions_root_ro = rotation_6d_to_matrix(motions[:,nj*6:nj*6+6].reshape((N,1,6))).reshape((N,1,3,3))#[B,T,6]
        motions_body = rotation_6d_to_matrix(motions[:,nj*6+6:nj*6+6+(nj-1)*6].reshape((N,nj-1,6))).reshape((N,nj-1,3,3))#[B,T,21*6]

        output = body_models(
            global_orient=motions_root_ro, 
            body_pose=motions_body,   
            transl=motions_transl
        ) 
        output = output.vertices.reshape((2,B, T,6890,3)) 

        #rending
        # for i in tqdm(range(T)):
        #     mesh_p1 = trimesh.Trimesh(
        #             vertices=output.detach().cpu().numpy()[0,0,i],#[T,6890,3]
        #             faces=body_models.faces,
        #             process=False,
        #         )
        #     mesh_p2 = trimesh.Trimesh(
        #             vertices=output.detach().cpu().numpy()[1,0,i],#[T,6890,3]
        #             faces=body_models.faces,
        #             process=False,
        #         )
        #     combined_mesh = trimesh.util.concatenate([mesh_p1, mesh_p2])
        #     combined_mesh.export(save_path + f'/ro6d_ply/{i}.ply')

        return output

    elif mode == 'smplx':
        nj = 55
        body_models = SMPLXLayer(model_path='./body_models/smplx',num_betas=100,gender='neutral',
                           joint_mapper=JointMapper(INTERX_TO_SMPLX),flat_hand_mean=False).to(device)
        motion_transl = motion[:,:3]# [B,T,3]
        motion_root_ro = rotation_6d_to_matrix(motion[:,nj*6:nj*6+6]).reshape((N,1,3,3))
        motion_body = rotation_6d_to_matrix( motion[:,nj*6+6:nj*6+(nj-1)*6]).reshape((N,nj-1,3,3))  
        motion_leye_ro = rotation_6d_to_matrix(motion[:,nj*6+nj*6:nj*6+nj*6+6]).reshape((N,1,3,3))
        motion_reye_ro = rotation_6d_to_matrix(motion[:,nj*6+nj*6+6:nj*6+nj*6+6+6]).reshape((N,1,3,3))
        motion_jaw_ro = rotation_6d_to_matrix(motion[:,nj*6+nj*6+6+6:nj*6+nj*6+6+6+6]).reshape((N,1,3,3))
        motion_left_hand_ro = rotation_6d_to_matrix(motion[:,nj*6+nj*6+6+6+6:nj*6+nj*6+6+6+6+10*6]).reshape((N,10,3,3))
        motion_right_hand_ro = rotation_6d_to_matrix(motion[:,nj*6+nj*6+6+6+6+10*6:nj*6+nj*6+6+6+6+10*6+10*6]).reshape((N,10,3,3))

        output = body_models(
            global_orient=motion_root_ro, 
            body_pose=motion_body,   
            transl=motion_transl,
            leye_pose=motion_leye_ro,
            reye_pose=motion_reye_ro,
            jaw_pose=motion_jaw_ro,
            left_hand_pose=motion_left_hand_ro,
            right_hand_pose=motion_right_hand_ro
        ) 
        output = output.vertices.reshape((B,T,6890,3))
        return output
    else:
        raise ValueError(f'Invalid mode: {mode}')
    