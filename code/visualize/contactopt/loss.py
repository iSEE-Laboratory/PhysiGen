import torch

L2=lambda x:torch.sum(x**2)
L1=lambda x:torch.sum(torch.abs(x))

def HOI_distance_loss(distance,hoi_pairs,device):
    """
    distance: (nf,n_object)
    hoi_pairs: (nf) of interacted object index,-1 means no interaction
    """
    
    nf=distance.shape[0]
    loss=torch.zeros((nf),device=device)
    lhas_interaction=(hoi_pairs!=-1)
    loss[lhas_interaction]=distance[lhas_interaction,hoi_pairs[lhas_interaction]]
    return torch.mean(loss)

def Loss_smooth_pose(pose):
    """
    pose:nf,nj,3
    """
    
    nf=pose.shape[0]
    pose_interp=pose.clone().detach()
    pose_interp[1:-1]=(pose_interp[1:-1]+pose_interp[:-2]+pose_interp[2:])/3
    loss=L2(pose[1:-1]-pose_interp[1:-1])
    return loss/(nf-2)

def Loss_reg_pose(pose):
    nf=pose.shape[0]
    loss=L2(pose)/nf
    return loss