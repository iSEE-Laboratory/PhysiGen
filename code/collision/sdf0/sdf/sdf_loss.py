import torch
import torch.nn as nn
import numpy as np

from sdf.sdf_model import SDF
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from bps_torch.bps import bps_torch
import time
import json

with open("/smpl_models/smpl_vert_segmentation.json", "r") as f:
    data = json.load(f)

# Merge all except leftHand and rightHand
body_index_wohand = []
l_hand = []
r_hand = []
for k, v in data.items():
    if k not in [ "leftHand", "rightHand",'leftHandIndex1', 'rightHandIndex1']: #
        body_index_wohand.extend(v)
    elif k in ['leftHandIndex1','leftHand']:
        l_hand.extend(v)
    elif k in [ 'rightHandIndex1','rightHand']:
        r_hand.extend(v)
    else:
        print('error')
# sort
body_index_wohand = sorted(set(body_index_wohand))



# with open("/smpl_models/smplx_vert_segmentation.json", "r") as f:
#     smplx_data = json.load(f)
# smplx_body_index_wohand = []
# smplx_l_hand = []
# smplx_r_hand = []
# for k, v in smplx_data.items():
#     if k not in [ "leftHand", "rightHand","leftHandIndex1", "rightHandIndex1"]: #
#         smplx_body_index_wohand.extend(v)
#     elif k in ['leftHandIndex1','leftHand']:
#         smplx_l_hand.extend(v)
#     elif k in [ 'rightHandIndex1','rightHand']:
#         smplx_r_hand.extend(v)
#     else:
#         print('error')
# # 去重排序
# smplx_body_index_wohand = sorted(set(smplx_body_index_wohand))


class SDFLoss(nn.Module):

    def __init__(self, faces, grid_size, weight=1, wohand = True, robustifier=None, debugging=False, device=None):
        super(SDFLoss, self).__init__()
        self.wohand = wohand
        self.sdf = SDF(grid_size=grid_size).to(device)
        person_points_num_wohand = 5366  #grid_size = 6890
        self.sdf_wohand = SDF(grid_size=person_points_num_wohand).to(device)
        self.register_buffer('faces', torch.tensor(faces.astype(np.int32), dtype=torch.int32).to(device).contiguous()) #

        

        self.grid_size = grid_size
        self.robustifier = robustifier
        self.debugging = debugging
        self.weight = weight
        self.bps = bps_torch(bps_type='random_uniform',
                n_bps_points=grid_size,
                radius=1.0,
                n_dims=3,
                custom_basis=None)

    
    # def to(self, device):
    #     self.device = device
    #     self.sdf = self.sdf.to(device)
    #     self.sdf_wohand = self.sdf_wohand.to(device)
    #     # self.bps = self.bps.to(device)
    #     return super().to(device)

    @torch.no_grad()
    def get_bounding_boxes(self, vertices1, vertices2):
        #vertices: (T, num_vertices, 3)
        # import ipdb;ipdb.set_trace()
        T = vertices1.shape[0]
        box1 = torch.zeros(T, 2, 3, device=vertices1.device)# Create a tensor with shape $(T, 2, 3)$ to store the bounding boxes for each person, containing the minimum and maximum values for $x, y, z$
        box2 = torch.zeros(T, 2, 3, device=vertices2.device)
        box1[:, 0, :] = vertices1.min(dim=1)[0] #[T,3]
        box1[:, 1, :] = vertices1.max(dim=1)[0]
        box2[:, 0, :] = vertices2.min(dim=1)[0]
        box2[:, 1, :] = vertices2.max(dim=1)[0]
        return box1, box2	
    

    @torch.no_grad()
    def check_overlap(self, box1, box2): #Determine if two bounding boxes overlap (Axis-Aligned Bounding Box collision detection
        #box1: (T, 2, 3)
        #box2: (T, 2, 3)
        T = box1.shape[0]
        overlapping_mask = torch.ones(T, device=box1.device, dtype=torch.bool)
        overlapping_mask &= ~( (box1[:, 0, 0] > box2[:, 1, 0]) |
                            (box2[:, 0, 0] > box1[:, 1, 0]) )
        overlapping_mask &= ~( (box1[:, 0, 1] > box2[:, 1, 1]) |
                            (box2[:, 0, 1] > box1[:, 1, 1]) )
        overlapping_mask &= ~( (box1[:, 0, 2] > box2[:, 1, 2]) |
                       (box2[:, 0, 2] > box1[:, 1, 2]) )
        return overlapping_mask



    def forward_metric(self, vertices1, vertices2, scale_factor=0.2, th=0.0):
        # vertices: (T, num_vertices, 3) T!=300
        

        T = vertices1.shape[0]

        # speed up
        box1, box2 = self.get_bounding_boxes(vertices1, vertices2) #[T, 2, 3]
        overlapping_mask = self.check_overlap(box1, box2) 
        # If no overlapping voxels return 0
        if overlapping_mask.sum() == 0: 
            print('there is no collision')
            tmp = torch.tensor(0., device=vertices1.device)
            return tmp,tmp,tmp,tmp,tmp,tmp  # ,tmp,tmp

        # Filter out the isolated boxes
        vertices1 = vertices1[overlapping_mask].contiguous()
        vertices2 = vertices2[overlapping_mask].contiguous()

        with torch.no_grad():

            phi1 = self.sdf(self.faces, vertices1, vertices2)  
            torch.cuda.synchronize()
            phi2 = self.sdf(self.faces, vertices2, vertices1)   
            torch.cuda.synchronize()

            if self.wohand:
                #remove hand 
                vertices1_hand_0 = vertices1.clone()
                vertices1_hand_0[:,l_hand] = vertices1_hand_0[:,l_hand[-1]].unsqueeze(1).clone()
                vertices1_hand_0[:,r_hand] = vertices1_hand_0[:,r_hand[-1]].unsqueeze(1).clone()
                vertices2_hand_0 = vertices2.clone()
                vertices2_hand_0[:,l_hand] = vertices2_hand_0[:,l_hand[-1]].unsqueeze(1).clone()
                vertices2_hand_0[:,r_hand] = vertices2_hand_0[:,r_hand[-1]].unsqueeze(1).clone()
                vertices1_wohand = vertices1[:,body_index_wohand]
                vertices2_wohand = vertices2[:,body_index_wohand]
                phi1_wohand = self.sdf_wohand(self.faces, vertices1_hand_0, vertices2_wohand)  
                phi2_wohand = self.sdf_wohand(self.faces, vertices2_hand_0, vertices1_wohand)


        pen12 = torch.where(phi1 < 0,  -phi1, 0.0)  # (T', V) 
        pen21 = torch.where(phi2 < 0,  -phi2, 0.0)
        sdf_dis  = (pen12.sum() + pen21.sum())   
        mask12 = phi1 <0         # bool (T, V)   
        mask21 = phi2 <0           # bool (T, V)    

        if self.wohand:
            pen12_wohand = torch.where(phi1_wohand< 0,  -phi1_wohand, 0.0)  #
            pen21_wohand = torch.where(phi2_wohand< 0,  -phi2_wohand, 0.0)  #
            sdf_dis_wohand  = (pen12_wohand.sum() + pen21_wohand.sum())  
            mask12_wohand = phi1_wohand<0
            mask21_wohand = phi2_wohand<0
            return sdf_dis, sdf_dis_wohand, mask12, mask21, mask12_wohand, mask21_wohand #, vertices1_wohand, vertices2_wohand
        else:
            return sdf_dis, None, mask12, mask21, None, None 
    def forward_metric_slow(self, vertices1, vertices2, scale_factor=0.2, th=0.0):
        # vertices: (T, num_vertices, 3) T!=300 
        T = vertices1.shape[0]

        with torch.no_grad():
            phi1 = self.sdf(self.faces, vertices1, vertices2)  
            torch.cuda.synchronize()
            phi2 = self.sdf(self.faces, vertices2, vertices1)   
            torch.cuda.synchronize()

            if self.wohand:
                vertices1_hand_0 = vertices1.clone()
                vertices1_hand_0[:,l_hand] = vertices1_hand_0[:,l_hand[-1]].unsqueeze(1).clone()
                vertices1_hand_0[:,r_hand] = vertices1_hand_0[:,r_hand[-1]].unsqueeze(1).clone()
                vertices2_hand_0 = vertices2.clone()
                vertices2_hand_0[:,l_hand] = vertices2_hand_0[:,l_hand[-1]].unsqueeze(1).clone()
                vertices2_hand_0[:,r_hand] = vertices2_hand_0[:,r_hand[-1]].unsqueeze(1).clone()
                vertices1_wohand = vertices1[:,body_index_wohand]
                vertices2_wohand = vertices2[:,body_index_wohand]
                phi1_wohand = self.sdf_wohand(self.faces, vertices1_hand_0, vertices2_wohand)  
                phi2_wohand = self.sdf_wohand(self.faces, vertices2_hand_0, vertices1_wohand)


        pen12 = torch.where(phi1 < 0,  -phi1, 0.0)  # (T', V) 
        pen21 = torch.where(phi2 < 0,  -phi2, 0.0)
        sdf_dis  = (pen12.sum() + pen21.sum())   
        mask12 = phi1 <0         # bool (T, V)    
        mask21 = phi2 <0           # bool (T, V)  

        if self.wohand:
            pen12_wohand = torch.where(phi1_wohand< 0,  -phi1_wohand, 0.0)  #
            pen21_wohand = torch.where(phi2_wohand< 0,  -phi2_wohand, 0.0)  #
            sdf_dis_wohand  = (pen12_wohand.sum() + pen21_wohand.sum())  
            mask12_wohand = phi1_wohand<0
            mask21_wohand = phi2_wohand<0
            return sdf_dis, sdf_dis_wohand, mask12, mask21, mask12_wohand, mask21_wohand #, vertices1_wohand, vertices2_wohand
        else:
            return sdf_dis, None, mask12, mask21, None, None 


    def forward_loss(self, vertices1, vertices2, scale_factor=0.2, th=0.0):
        # vertices: (T, num_vertices, 3) T!=300
        T = vertices1.shape[0]

        vertices1 = vertices1.contiguous()
        vertices2 = vertices2.contiguous()
    
        vertices1_q = vertices1[:,:self.grid_size] #.clone()
        vertices2_q = vertices2[:,:self.grid_size] #.clone()

        phi1 = self.sdf(self.faces, vertices1, vertices2_q)   
        phi2 = self.sdf(self.faces, vertices2, vertices1_q)   


        pen12 = torch.where(phi1 < 0,  -phi1, 0.0)  # (T', V) 
        pen21 = torch.where(phi2 < 0,  -phi2, 0.0)

        loss  = (pen12.sum() + pen21.sum()) * self.weight / vertices1.shape[0] #计算loss

        mask12 = phi1 <0         # bool (T, V)    
        mask21 = phi2 <0           # bool (T, V)    

        return loss, mask12, mask21, vertices1_q, vertices2_q