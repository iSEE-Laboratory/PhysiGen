import numpy as np
import torch
from tqdm import tqdm

from in2in.visualize.smplifyx.config import *
from in2in.visualize.smplifyx.lbfgs import LBFGS
from in2in.visualize.smplifyx.loss import *
from in2in.visualize.utils.geometry import batch_rodrigues
from in2in.visualize.utils.limbs import OPTITRACK_LIMBS

def multi_stage_optimize(params,body_models,kp3ds, mode, device):
    """
    kp3ds: nf,nj,4

    Multi-stage optimizing
    1. Shape
    2. Global orient and transl
    3. Poses
    """
    # Use pre-computed shape

    # optimize RT
    params = optimize_pose(params,body_models,kp3ds,
                         OPT_RT=True, mode=mode, device=device)
    # optimize body poses
    params = optimize_pose(params,body_models,kp3ds,
                         OPT_RT=True, OPT_POSE=True, mode=mode, device=device)
    
    # optimize hand poses
    if mode == 'smplh':
        # params = optimize_pose(params,body_models,kp3ds,
        #                     OPT_RT=False,OPT_POSE=True,OPT_HAND=True, mode=mode)
        pass
    elif mode == 'smplx':
        params = optimize_pose(params,body_models,kp3ds,
                            OPT_RT=False,OPT_POSE=True,OPT_HAND=True, mode=mode, device=device)
    
    return params

# 定义优化人体参数的函数，包含姿态、平移、手部、表情等参数
def optimize_pose(params, body_models, kp3ds, mode,
                  OPT_RT=False, OPT_POSE=False,
                  OPT_HAND=False, OPT_EXPR=False, device=None):
    nf = kp3ds.shape[0]  # 获取帧数，即关键点3D数据的时间维度
    loss_dict = []       # 存储使用的损失函数名称
    opt_params = []      # 存储要优化的参数列表
    if OPT_RT:# 如果要优化全局旋转和平移（RT）
        
        loss_dict+=[
            'k3d','reg_pose' ,'smooth_pose','smooth_body'# 添加对应的损失项
        ]
        opt_params+=[params['global_orient'],params['transl']] # 添加需要优化的旋转和平移参数
        loss_weight=OPTIMIZE_RT # 设置当前优化任务的损失权重
        desc='Optimizing RT...' # 当前任务描述

    if OPT_POSE:
        opt_params+=[params['body_pose']] # 添加身体姿态参数
        loss_weight=OPTIMIZE_POSES
        desc='Optimizing Body pose...'

    if OPT_HAND:# 如果要优化手部姿态
        loss_dict+=[
            'k3d_hand','reg_hand','smooth_hand','k3d','reg_pose' ,'smooth_pose','smooth_body'
        ]
        opt_params+=[params['lhand_pose'],params['rhand_pose']] # add wrist to optimize
        loss_weight=OPTIMIZE_HAND
        desc='Optimizing Hand...'

    if OPT_EXPR:# 如果要优化面部表情
        loss_dict+=[
            'k3d_face','reg_head','reg_expr','smooth_head'
        ]
        opt_params+=[params['jaw_pose'],params['leye_pose'],
                     params['reye_pose'],params['expression']]
        loss_weight=OPTIMIZE_EXPR
        desc='Optimizing Expression...'

    optimizer = LBFGS(opt_params,line_search_fn='strong_wolfe',max_iter=30) ## 定义优化器，使用 L-BFGS 算法

    def closure(debug=False): # 定义闭包函数，用于计算损失和梯度
        optimizer.zero_grad()
        if mode == 'smplh':
            axis_angle = torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            # params['leye_pose'][:,None,:],
                            # params['reye_pose'][:,None,:],
                            # params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # nf*52*2,3
            rot_mat = batch_rodrigues(axis_angle).reshape((nf,-1,3,3)) # nf,nj,3,3
            ## 使用 SMPL-H 模型生成3D关节点

            # print("rot_mat.shape", rot_mat.shape)
            # print("params['transl'].shape", params['transl'].shape)
            # print("params['global_orient'].shape", params['global_orient'].shape)
            # print("params['body_pose'].shape", params['body_pose'].shape)
            # print("params['lhand_pose'].shape", params['lhand_pose'].shape)
            # print("params['rhand_pose'].shape", params['rhand_pose'].shape)

            out_kp3d = body_models(
                global_orient=rot_mat[:,0,...],
                body_pose=rot_mat[:,1:22,...],
                left_hand_pose=rot_mat[:,22:37,...],
                right_hand_pose=rot_mat[:,37:52,...],
                transl=params['transl'] 
            ).joints # nf,nj(JOINT_MAPPER),3     22+30
            # import ipdb; ipdb.set_trace()
        elif mode == 'smplx':
            axis_angle = torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['leye_pose'][:,None,:],
                            params['reye_pose'][:,None,:],
                            params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)).to(device) # nf*55,3
            rot_mat=batch_rodrigues(axis_angle).reshape((nf,-1,3,3)) # nf,55,3,3
            out_kp3d=body_models(
                global_orient=rot_mat[:,0,...],
                body_pose=rot_mat[:,1:22,...],
                leye_pose=rot_mat[:,22,...],
                reye_pose=rot_mat[:,23,...],
                jaw_pose=rot_mat[:,24,...],
                left_hand_pose=rot_mat[:,25:40,...],
                right_hand_pose=rot_mat[:,40:55,...],
                transl=params['transl']
            ).joints # nf,nj(JOINT_MAPPER),3

        ## 计算所有定义的损失项

        final_loss_dict={loss_name : get_loss(loss_name,kp3ds, out_kp3d, params, mode) 
                         for loss_name in loss_dict }
        loss = sum([final_loss_dict[key]*loss_weight[key]
                  for key in loss_dict])
        if not debug:
            loss.backward()
            return loss
        else:
            return final_loss_dict
    
    final_loss = run_fitting(optimizer,closure,opt_params,desc)# 调用优化器运行拟合过程

    final_loss_dict = closure(debug=True)  # 最后再运行一次 closure 获取最终损失（不反向传播）
    # for key in final_loss_dict.keys():
    #     print("%s : %f"%(key,final_loss_dict[key].item()))

    return params
        
def optimize_shape(params,body_models,kp3ds,mode):
    nf=kp3ds.shape[0]

    start=torch.tensor(np.array(OPTITRACK_LIMBS)[:,0],device='cuda')
    end=torch.tensor(np.array(OPTITRACK_LIMBS)[:,1],device='cuda')
    start_kp3d=torch.index_select(kp3ds,1,start) # nf,nlimbs,4
    end_kp3d=torch.index_select(kp3ds,1,end)
    # nf,nlimbs,1
    limb_length=torch.norm(start_kp3d[...,:3]-end_kp3d[...,:3],dim=2,keepdim=True) 
    # nf,nlimbs,1
    limb_conf=torch.minimum(start_kp3d[...,3],end_kp3d[...,3])[...,None]

    opt_params = [params['betas']]
    optimizer = LBFGS(opt_params,line_search_fn='strong_wolfe',max_iter=30) #100

    def closure(debug=False):
        optimizer.zero_grad()
        if mode == 'smplh':
            axis_angle=torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            # params['leye_pose'][:,None,:],
                            # params['reye_pose'][:,None,:],
                            # params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # nf*55,3
            rot_mat=batch_rodrigues(axis_angle).reshape((nf,-1,3,3)) # nf,55,3,3
            out_kp3d=body_models(
                transl=params['transl'],
                global_orient=rot_mat[:,0,...],
                body_pose=rot_mat[:,1:22,...],
                # leye_pose=rot_mat[:,22,...],
                # reye_pose=rot_mat[:,23,...],
                # jaw_pose=rot_mat[:,24,...],
                left_hand_pose=rot_mat[:,22:37,...],
                right_hand_pose=rot_mat[:,37:52,...],
            ).joints # nf,nj(JOINT_MAPPER),3
        elif mode == 'smplx':
            axis_angle=torch.cat([params['global_orient'][:,None,:],
                            params['body_pose'],
                            params['leye_pose'][:,None,:],
                            params['reye_pose'][:,None,:],
                            params['jaw_pose'][:,None,:],
                            params['lhand_pose'],
                            params['rhand_pose']
                            ],axis=1).reshape((-1,3)) # nf*55,3
            rot_mat=batch_rodrigues(axis_angle).reshape((nf,-1,3,3)) # nf,55,3,3
            out_kp3d=body_models(
                transl=params['transl'],
                global_orient=rot_mat[:,0,...],
                body_pose=rot_mat[:,1:22,...],
                leye_pose=rot_mat[:,22,...],
                reye_pose=rot_mat[:,23,...],
                jaw_pose=rot_mat[:,24,...],
                left_hand_pose=rot_mat[:,25:40,...],
                right_hand_pose=rot_mat[:,40:55,...]
            ).joints # nf,nj(JOINT_MAPPER),3

        out_start_kp3d=torch.index_select(out_kp3d,1,start)
        out_end_kp3d=torch.index_select(out_kp3d,1,end)
        out_dist=(out_start_kp3d[...,:3]-out_end_kp3d[...,:3]).detach()
        out_dist_norm=torch.norm(out_dist,dim=2,keepdim=True) 
        out_dist_normalized=out_dist/(out_dist_norm+1e-4)
        err=(out_start_kp3d[...,:3]-out_end_kp3d[...,:3])\
            -out_dist_normalized*limb_length
        loss_dict={
            'shape3d':Loss_shape3d(err,limb_conf,nf),
            'reg_shape':Loss_reg_shape(params['betas'])
        }
        loss_weight=OPTIMIZE_SHAPE
        loss=sum([loss_dict[key]*loss_weight[key] 
                  for key in loss_dict.keys()])
    
        if not debug:
            loss.backward()
            return loss
        else:
            return loss_dict
    final_loss=run_fitting(optimizer,closure,opt_params,"Optimizing shape...")
    loss_dict=closure(True)
    # for key in loss_dict.keys():
    #     print("%s : %f"%(key,loss_dict[key].item()))
    return params

def run_fitting(optimizer,closure,opt_params,desc,maxiters=50, ftol=1e-9):
    prev_loss = None # 记录上一轮的损失，用于判断收敛
    require_grad(opt_params,True)  # 启用需要优化参数的梯度计算
    for iter in tqdm(range(maxiters),desc=desc):
        loss = optimizer.step(closure) # 运行优化器一步，closure 会返回当前损失并计算梯度
        if torch.isnan(loss).sum() > 0:# 如果损失出现 NaN（数值不合法），终止优化
                print('NaN loss value, stopping!')
                break

        if torch.isinf(loss).sum() > 0:# 如果损失为 Inf（无穷大），终止优化
            print('Infinite loss value, stopping!')
            break

        if iter>0 and prev_loss is not None and ftol>0:# 如果已迭代至少一次并设置了收敛阈值，判断损失相对变化是否小于阈值
            loss_rel_change = rel_change(prev_loss,loss.item())
            if loss_rel_change <= ftol: # 如果小于设定的阈值，认为已经收敛，提前结束
                break
        prev_loss = loss.item()# 更新上一轮损失值

    require_grad(opt_params,False)# 优化结束后关闭参数的梯度计算（节省资源）
    return prev_loss

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

def require_grad(opt_params,flag=False):
    for param in opt_params:
        param.requires_grad=flag