import torch
import numpy as np
from visualize.joints_to_smpl import joints_smplifyx_metric
from visualize.ro6d_to_smpl import ro6d_smplifyx_metric
from sdf.sdf_loss import SDFLoss

def coll_metric_mesh(body_models, motion1, motion2, motion_len, device):
    # motion: (300,22,3)
    motion1 = motion1[:motion_len].reshape(motion_len, 22 ,3)
    motion2 = motion2[:motion_len].reshape(motion_len, 22 ,3)
    motion1 = torch.tensor(motion1).float().to(device)
    motion2 = torch.tensor(motion2).float().to(device)
    vertices_p1 = joints_smplifyx_metric_single(motion1, save_path=None, body_models=body_models.to(device), mode='smplh', device=device)#[T,N,3]
    vertices_p2 = joints_smplifyx_metric_single(motion2, save_path=None, body_models=body_models.to(device), mode='smplh', device=device) #[T,N,3]
    vertices_p1 = torch.tensor(vertices_p1).to(device) #[T,N,3]
    vertices_p2 = torch.tensor(vertices_p2).to(device)
    loss_fn = SDFLoss(body_models.faces, grid_size=vertices_p1.shape[1], device=device)
    coll_dis, coll_dis_wohand, mask12,mask21, mask12_wohand, mask21_wohand = loss_fn.forward_metric(vertices_p1, vertices_p2)
    
    if coll_dis == 0:
        coll_frame = 0.0
    else:
        coll_frame = mask12.any(dim=1).sum()
    if coll_dis_wohand == 0:
        coll_frame_wohand = 0.0
    else:
        coll_frame_wohand = mask12_wohand.any(dim=1).sum()

    return coll_dis/motion_len , coll_frame/motion_len, coll_dis_wohand/motion_len, coll_frame_wohand/motion_len