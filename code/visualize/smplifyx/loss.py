import torch

L2=lambda x:torch.sum(x**2)
L1=lambda x:torch.sum(torch.abs(x))
Gmof=lambda x:torch.sum(gmof(x**2,0.04))

from in2in.visualize.utils.mapping import OPTITRACK_BODY_smplh, OPTITRACK_BODY_smplx, OPTITRACK_HAND_smplh, OPTITRACK_HAND_smplx

def gmof(squared_res, sigma_squared):
    """
    Geman-McClure error function
    """
    return (sigma_squared * squared_res) / (sigma_squared + squared_res)

def get_loss(loss_name,kp3ds,out_kp3ds,params, mode):
    if loss_name=='k3d':
        return Loss_k3d(kp3ds,out_kp3ds,part='body', mode=mode)
    elif loss_name=='k3d_hand':
        return Loss_k3d(kp3ds,out_kp3ds,part='hand',norm='l2', mode=mode)
    elif loss_name=='k3d_face': 
        return Loss_k3d(kp3ds,out_kp3ds,part='face',norm='l1', mode=mode)
    elif loss_name=='smooth_body':
        return Loss_smooth_body(out_kp3ds,part='body', mode=mode)
    elif loss_name=='smooth_hand':
        return Loss_smooth_body(out_kp3ds,part='hand', mode=mode)
    elif loss_name=='smooth_pose':
        return Loss_smooth_pose(params['body_pose'])
    elif loss_name=='smooth_head':
        return Loss_smooth_pose(
            torch.cat([params['jaw_pose'],
                       params['leye_pose'],
                       params['reye_pose']],dim=1))
    elif loss_name=='reg_pose':
        return Loss_reg_pose(params['body_pose'],part='body')
    elif loss_name=='reg_hand':
        return Loss_reg_pose(params['body_pose'],part='hand')
    elif loss_name=='reg_head':
        return Loss_reg_pose(params['body_pose'],part='head')
    elif loss_name=='reg_expr':
        return Loss_reg_pose(params['body_pose'],part='expr')
    else:
        raise NotImplementedError
    
def Loss_shape3d(err,limb_conf,nf):
    return torch.sum(err**2*limb_conf)/nf

def Loss_reg_shape(betas):
    return torch.sum(betas**2)

def Loss_k3d(kp3ds,out_kp3ds,part='body',norm='l2', mode='smplx'):
    nf=kp3ds.shape[0]
    if norm=='l1':
        norm_func=L1
    elif norm=='l2':
        norm_func=L2
    elif norm=='gmof':
        norm_func=Gmof
    else:
        raise NotImplementedError
    
    if part=='body':
        if mode == 'smplh':
            diff_square=(kp3ds[:,OPTITRACK_BODY_smplh,:3]-out_kp3ds[:,OPTITRACK_BODY_smplh,:3])*kp3ds[:,OPTITRACK_BODY_smplh,3][...,None]
        elif mode == 'smplx':
            diff_square=(kp3ds[:,OPTITRACK_BODY_smplx,:3]-out_kp3ds[:,OPTITRACK_BODY_smplx,:3])*kp3ds[:,OPTITRACK_BODY_smplx,3][...,None]
    elif part=='hand':
        if mode == 'smplh':
            diff_square=(kp3ds[:,OPTITRACK_HAND_smplh,:3]-out_kp3ds[:,OPTITRACK_HAND_smplh,:3])*kp3ds[:,OPTITRACK_HAND_smplh,3][...,None]
        elif mode == 'smplx':
            diff_square=(kp3ds[:,OPTITRACK_HAND_smplx,:3]-out_kp3ds[:,OPTITRACK_HAND_smplx,:3])*kp3ds[:,OPTITRACK_HAND_smplx,3][...,None]
    elif part=='face':
        diff_square=(kp3ds[:,25+42:,:3]-out_kp3ds[:,25+42:,:3])*kp3ds[:,25+42:,3][...,None]
    
    return norm_func(diff_square)/nf

def Loss_smooth_body(out_kp3ds,part='body', mode='smplx'):
    nf=out_kp3ds.shape[0]
    if part=='body':
        if mode == 'smplh':
            kp3ds_est=out_kp3ds[:,OPTITRACK_BODY_smplh]
        elif mode == 'smplx':
            kp3ds_est=out_kp3ds[:,OPTITRACK_BODY_smplx]
    elif part=='hand':
        if mode == 'smplh':
            kp3ds_est=out_kp3ds[:,OPTITRACK_HAND_smplh]
        elif mode == 'smplx':
            kp3ds_est=out_kp3ds[:,OPTITRACK_HAND_smplx]

    kp3ds_interp=kp3ds_est.clone().detach()
    kp3ds_interp[1:-1]=(kp3ds_interp[:-2]+kp3ds_interp[2:])/2
    loss=L2(kp3ds_est[1:-1]-kp3ds_interp[1:-1])

    return loss/(nf-2)

def Loss_smooth_pose(pose):
    """
    pose:nf,nj,3
    """
    
    nf=pose.shape[0]
    pose_interp=pose.clone().detach()
    pose_interp[1:-1]=(pose_interp[1:-1]+pose_interp[:-2]+pose_interp[2:])/3
    loss=L2(pose[1:-1]-pose_interp[1:-1])
    return loss/(nf-2)

def Loss_reg_pose(pose,part="body"):
    nf=pose.shape[0]
    if part in ['body','hand','head']:
        loss=L2(pose)/nf
    elif part=='expr':
        loss=L2(pose)
    return loss




