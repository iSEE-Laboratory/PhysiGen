
import torch
import numpy as np
from visualize.joints_to_smpl import joints_smplifyx_metric
from visualize.ro6d_to_smpl import ro6d_smplifyx_loss
from sdf.sdf_loss import SDFLoss
import time


def compute_coll_loss_mesh(prediction_, mask, param1, param2, body_models, points_sampled_on_mesh=128):
    # prediction_ : [batch, T, 2, 268]
    # mask : [batch, T, 2, 1]
    B,T = prediction_.shape[:2]
    motion1 = prediction_[:,:,0].reshape(B,T, 268)
    motion2 = prediction_[:,:,1].reshape(B,T, 268)
    device = motion1.device
    # get mesh
    vertices_ro6d = ro6d_smplifyx_loss(motion1, motion2, save_path='tmp/ro6d_ply', body_models=body_models.to(device), mode='smplh') #(2,B, T,6890,3)
    vertices_p1 = vertices_ro6d[0].to(device) #[B,T,N,3]
    vertices_p2 = vertices_ro6d[1].to(device)  #[B,T,N,3]
    loss_fn = SDFLoss(body_models.faces, grid_size=points_sampled_on_mesh, device=device) #6890

    loss = torch.tensor(0.0).to(device)


    len_all = 0
    for i in range(B):
        start_time_i = time.time()
        motion_len1 = int(mask[i,:,0].sum())
        motion_len2 = int(mask[i,:,1].sum())
        # compute sdf_loss
        coll_dis_ro6d, _,_,_,_  = loss_fn.forward_loss(vertices_p1[i][:motion_len1], vertices_p2[i][:motion_len2])
        loss += coll_dis_ro6d.sum()
        len_all += motion_len1
    


    return loss/(B*len_all*2) # Return the collision distance per person per frame










