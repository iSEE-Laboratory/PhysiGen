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
import warnings
warnings.filterwarnings("ignore", message="You are using a SMPL+H model")

from in2in.visualize.smplifyx.optimize import *
from in2in.visualize.utils.io import write_smplx
#from in2in.visualize.utils.mapping import (INTERX_TO_SMPLX, INTERX_TO_SMPLH, JointMapper)
from in2in.visualize.utils.torch_utils import *
from ipdb import set_trace
from in2in.visualize.utils.geometry import batch_rodrigues
import trimesh
import h5py


def joints_smplifyx_metric(motion1, motion2, save_path, body_models,mode,device):
    #motion [T, 22, 3] 
    motion_len = motion1.shape[0]
    motions = torch.cat((motion1, motion2), dim=0)
    N = motion_len*2
    motions = motions.reshape((N, -1)).detach().cpu().numpy()
    
    if mode == 'smplh':
        nj = 22
        joint_positions = motions[:, :nj*3].reshape((N,nj,3))
        # body_models = SMPLHLayer(model_path='./smpl_models/smplh',num_betas=10,gender='neutral',
        #                 joint_mapper=JointMapper(INTERX_TO_SMPLH),flat_hand_mean=False,ext='npz').to('cuda')

        joint_positions = np.concatenate((joint_positions, np.zeros((N,30,3))), axis=1)
        
        # create params to be optimized,
        params={
            'body_pose':np.zeros((N,21,3)),
            'lhand_pose':np.zeros((N,15,3))  ,
            'rhand_pose':np.zeros((N,15,3)),
            'global_orient':np.zeros((N,3)),
            'transl':np.zeros((N,3)),
        }

    elif mode == 'smplx':
        body_models = SMPLXLayer(model_path='./smpl_models/smplx',num_betas=10,gender='neutral',
                           joint_mapper=JointMapper(INTERX_TO_SMPLX),flat_hand_mean=False).to(device)

        # create params to be optimized,
        params={
            'body_pose':np.zeros((N,21,3)),
            'lhand_pose':np.zeros((N,15,3)),
            'rhand_pose':np.zeros((N,15,3)),
            'jaw_pose':np.zeros((N,3)),
            'leye_pose':np.zeros((N,3)),
            'reye_pose':np.zeros((N,3)),
            'global_orient':np.zeros((N,3)),
            'transl':np.zeros((N,3)),
        }
    
    
    params = init_params(params, body_models, N,device) #move to cuda
    nj = joint_positions.shape[1]
    # add confidence 1.0
    joint_positions = np.concatenate([joint_positions, np.ones((N,nj,1))],axis=-1)
    # however, make the confidence of NaN to 0.0
    nan_joints = np.isnan(joint_positions).any(axis=-1)
    joint_positions[nan_joints]=0.0
    joint_positions = torch.tensor(joint_positions, dtype=torch.float32, device=device)

    # SMPLify-X optimization
    params = multi_stage_optimize(params, body_models, joint_positions, mode=mode, device=device)


    # visualize the result
    if mode == 'smplh':
        axis_angle = torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # T*52,3
        rot_mat = batch_rodrigues(axis_angle).reshape((N,-1,3,3)) # T,52,3,3
        ## 使用 SMPL-H 模型生成3D关节点
        output = body_models(
            global_orient=rot_mat[:,0,...],
            body_pose=rot_mat[:,1:22,...],
            left_hand_pose=rot_mat[:,22:37,...],
            right_hand_pose=rot_mat[:,37:52,...],
            transl=params['transl']
        ) # N,nj(JOINT_MAPPER),3
    elif mode == 'smplx':
        axis_angle=torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['leye_pose'][:,None,:],
                            params['reye_pose'][:,None,:],
                            params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # N*55,3
        rot_mat = batch_rodrigues(axis_angle).reshape((N,-1,3,3)) # N,55,3,3
        output = body_models(
            global_orient=rot_mat[:,0,...],
            body_pose=rot_mat[:,1:22,...],
            leye_pose=rot_mat[:,22,...],
            reye_pose=rot_mat[:,23,...],
            jaw_pose=rot_mat[:,24,...],
            left_hand_pose=rot_mat[:,25:40,...],
            right_hand_pose=rot_mat[:,40:55,...],
            transl=params['transl']
        ) # N,nj(JOINT_MAPPER),3

        # for i in tqdm(range(T)):
    
    
    output = output.vertices.reshape((2,motion_len,6890,3)).detach().cpu().numpy()

    # 可视化检查
    if save_path != None:
        for i in tqdm(range(motion_len)):
            mesh_p1 = trimesh.Trimesh(
                    vertices=output[0,i],#[6890,3]
                    faces=body_models.faces,
                    process=False,
                )
            mesh_p2 = trimesh.Trimesh(
                    vertices=output[1,i],#[6890,3]
                    faces=body_models.faces,
                    process=False,
                )
            combined_mesh = trimesh.util.concatenate([mesh_p1, mesh_p2])
            combined_mesh.export('./tmp' + f'/joints_ply/{i}.ply')


    
    return output


def joints_smplifyx_metric_single(motion, save_path, body_models,mode,device, name='none'):
    #motion T, 22, 3
    motion_len = motion.shape[0]
    N = motion_len
    motion = motion.reshape((N, -1)).detach().cpu().numpy()
    
    if mode == 'smplh':
        nj = 22
        joint_position = motion[:, :nj*3].reshape((N,nj,3))
        # body_models = SMPLHLayer(model_path='./smpl_models/smplh',num_betas=10,gender='neutral',
        #                 joint_mapper=JointMapper(INTERX_TO_SMPLH),flat_hand_mean=False,ext='npz').to('cuda')

        joint_position = np.concatenate((joint_position, np.zeros((N,30,3))), axis=1)
        
        # create params to be optimized,
        params={
            'body_pose':np.zeros((N,21,3)),
            'lhand_pose':np.zeros((N,15,3))  ,
            'rhand_pose':np.zeros((N,15,3)),
            'global_orient':np.zeros((N,3)),
            'transl':np.zeros((N,3)),
        }

    elif mode == 'smplx':
        body_models = SMPLXLayer(model_path='./smpl_models/smplx',num_betas=10,gender='neutral',
                           joint_mapper=JointMapper(INTERX_TO_SMPLX),flat_hand_mean=False).to(device)

        # create params to be optimized,
        params={
            'body_pose':np.zeros((N,21,3)),
            'lhand_pose':np.zeros((N,15,3)),
            'rhand_pose':np.zeros((N,15,3)),
            'jaw_pose':np.zeros((N,3)),
            'leye_pose':np.zeros((N,3)),
            'reye_pose':np.zeros((N,3)),
            'global_orient':np.zeros((N,3)),
            'transl':np.zeros((N,3)),
        }
    
    
    params = init_params(params, body_models, N,device) #move to cuda
    nj = joint_position.shape[1]
    # add confidence 1.0
    joint_position = np.concatenate([joint_position, np.ones((N,nj,1))],axis=-1)
    # however, make the confidence of NaN to 0.0
    nan_joints = np.isnan(joint_position).any(axis=-1)
    joint_position[nan_joints]=0.0
    joint_position = torch.tensor(joint_position, dtype=torch.float32, device=device)

    # SMPLify-X optimization
    params = multi_stage_optimize(params, body_models, joint_position, mode=mode, device=device)


    # visualize the result
    if mode == 'smplh':
        axis_angle = torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # T*52,3
        rot_mat = batch_rodrigues(axis_angle).reshape((N,-1,3,3)) # T,52,3,3
        ## 使用 SMPL-H 模型生成3D关节点
        output = body_models(
            global_orient=rot_mat[:,0,...],
            body_pose=rot_mat[:,1:22,...],
            left_hand_pose=rot_mat[:,22:37,...],
            right_hand_pose=rot_mat[:,37:52,...],
            transl=params['transl']
        ) # N,nj(JOINT_MAPPER),3
    elif mode == 'smplx':
        axis_angle=torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['leye_pose'][:,None,:],
                            params['reye_pose'][:,None,:],
                            params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # N*55,3
        rot_mat = batch_rodrigues(axis_angle).reshape((N,-1,3,3)) # N,55,3,3
        output = body_models(
            global_orient=rot_mat[:,0,...],
            body_pose=rot_mat[:,1:22,...],
            leye_pose=rot_mat[:,22,...],
            reye_pose=rot_mat[:,23,...],
            jaw_pose=rot_mat[:,24,...],
            left_hand_pose=rot_mat[:,25:40,...],
            right_hand_pose=rot_mat[:,40:55,...],
            transl=params['transl']
        ) # N,nj(JOINT_MAPPER),3

        # for i in tqdm(range(T)):
    
    
    output = output.vertices.reshape((motion_len,6890,3)).detach().cpu().numpy()

    # 可视化检查
    if save_path is not None:
        for i in tqdm(range(motion_len)):
            mesh_p1 = trimesh.Trimesh(
                    vertices=output[i],#[T,6890,3]
                    faces=body_models.faces,
                    process=False,
            )
            mesh_p1.export(save_path + f'/mesh/{name}_{i}.ply')


    
    return output







def joints_smplifyx(motion1, motion2, save_path, mode):
    # motion1: B,T,268 
    B,T = motion1.shape[:2]
    motions = torch.cat((motion1, motion2), dim=0)
    N = B*T*2
    motions = motions.reshape((N, -1)).detach().cpu().numpy()

    # create body models
    if mode == 'smplh':
        nj = 22
        joint_positions = motions[:, :nj*3].reshape((N,nj,3))
        body_models = SMPLHLayer(model_path='./smpl_models/smplh',num_betas=10,gender='neutral',
                           joint_mapper=JointMapper(INTERX_TO_SMPLH),flat_hand_mean=False,ext='npz').to('cuda')

        joint_positions = np.concatenate((joint_positions, np.zeros((N,30,3))), axis=1)
        
        # create params to be optimized,
        params={
            'body_pose':np.zeros((N,21,3)),
            'lhand_pose':np.zeros((N,15,3)),
            'rhand_pose':np.zeros((N,15,3)),
            'global_orient':np.zeros((N,3)),
            'transl':np.zeros((N,3)),
        }

    elif mode == 'smplx':
        body_models = SMPLXLayer(model_path='./smpl_models/smplx',num_betas=10,gender='neutral',
                           joint_mapper=JointMapper(INTERX_TO_SMPLX),flat_hand_mean=False).to('cuda')

        # create params to be optimized,
        params={
            'body_pose':np.zeros((N,21,3)),
            'lhand_pose':np.zeros((N,15,3)),
            'rhand_pose':np.zeros((N,15,3)),
            'jaw_pose':np.zeros((N,3)),
            'leye_pose':np.zeros((N,3)),
            'reye_pose':np.zeros((N,3)),
            'global_orient':np.zeros((N,3)),
            'transl':np.zeros((N,3)),
        }
    
    
    params = init_params(params, body_models, N, device) #move to cuda
    nj = joint_positions.shape[1]
    # add confidence 1.0
    joint_positions = np.concatenate([joint_positions, np.ones((N,nj,1))],axis=-1)
    # however, make the confidence of NaN to 0.0
    nan_joints = np.isnan(joint_positions).any(axis=-1)
    joint_positions[nan_joints]=0.0
    # convert joint_positions to tensor
    joint_positions = torch.tensor(joint_positions, dtype=torch.float32, device='cuda')

    # SMPLify-X optimization
    params = multi_stage_optimize(params, body_models, joint_positions, mode=mode)


    # visualize the result
    if mode == 'smplh':
        axis_angle = torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # T*52,3
        rot_mat = batch_rodrigues(axis_angle).reshape((N,-1,3,3)) # T,52,3,3
        ## 使用 SMPL-H 模型生成3D关节点
        output = body_models(
            global_orient=rot_mat[:,0,...],
            body_pose=rot_mat[:,1:22,...],
            left_hand_pose=rot_mat[:,22:37,...],
            right_hand_pose=rot_mat[:,37:52,...],
            transl=params['transl']
        ) # N,nj(JOINT_MAPPER),3
    elif mode == 'smplx':
        axis_angle=torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['leye_pose'][:,None,:],
                            params['reye_pose'][:,None,:],
                            params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # N*55,3
        rot_mat = batch_rodrigues(axis_angle).reshape((N,-1,3,3)) # N,55,3,3
        output = body_models(
            global_orient=rot_mat[:,0,...],
            body_pose=rot_mat[:,1:22,...],
            leye_pose=rot_mat[:,22,...],
            reye_pose=rot_mat[:,23,...],
            jaw_pose=rot_mat[:,24,...],
            left_hand_pose=rot_mat[:,25:40,...],
            right_hand_pose=rot_mat[:,40:55,...],
            transl=params['transl']
        ) # N,nj(JOINT_MAPPER),3


    output = output.vertices.reshape((2,B,T,6890,3)).detach().cpu().numpy()

    # rending 
    # for i in tqdm(range(T)):
    #     mesh_p1 = trimesh.Trimesh(
    #             vertices=output[0,0,i],#[T,6890,3]
    #             faces=body_models.faces,
    #             process=False,
    #         )
    #     mesh_p2 = trimesh.Trimesh(
    #             vertices=output[1,0,i],#[T,6890,3]
    #             faces=body_models.faces,
    #             process=False,
    #         )
    #     combined_mesh = trimesh.util.concatenate([mesh_p1, mesh_p2])
    #     combined_mesh.export(save_path + f'/joints_ply/{i}.ply')

    return output

    parser = argparse.ArgumentParser()
    parser.add_argument('--npy',type=str,default='./1.npy')
    parser.add_argument('--save_path',type=str,default='./tmp')
    parser.add_argument('--dataset',type=str,default='interhuman')


    # parser.add_argument('--npy',type=str,default='/home/leinan/dataset/inter_X/interx_humanml3d/intergen/motions/G053T006A029R024.npy')
    # parser.add_argument('--save_path',type=str,default='./mesh/tmp')
    # parser.add_argument('--dataset',type=str,default='interx')
    
    args = parser.parse_args()

    # do smplify
    if args.dataset == 'interhuman':
        # load 3D joints positions
        num_joints = 22 #55
        joint_positions_all = load_joint_positions(args.npy, num_joints)#B*T*2,J,3
        smplh_res = smplifyx(joint_positions_all, args.save_path, mode='smplh')

        np.save(args.save_path + 'param/1', np.stack(smplh_res))
    elif args.dataset == 'interx':
        num_joints = 55
        joint_positions_all = load_joint_positions_interx_gt(args.npy, num_joints)
        smplh_res = smplifyx(joint_positions_all, args.save_path, mode='smplx')
        np.save(args.save_path+ '/1', np.stack(smplx_res))