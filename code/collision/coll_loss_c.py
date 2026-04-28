
import torch
import torch.nn as nn
import torch.nn.functional as F
from .param import box_smplx,t2m_kinematic_chain
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


def show_video_c(motion1, motion2, points1, points2, coll_index1, coll_index2,figsize=(10, 10), fps=120, radius=4):
    title_sp = '1'
    frame_number = motion1.shape[0]
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle('tmp', fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    fig = plt.figure(figsize=(10,10))
    ax = p3.Axes3D(fig)
    init()

    def update(index):
        #print("index",index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90) #0
        ax.dist = 15#7.5
        #         ax =
        plot_xzPlane(-3, 3, 0, -3, 3)
        ax.scatter(points1[index][:,0].detach().cpu().numpy(), 
                   points1[index][:,1].detach().cpu().numpy(),
                   points1[index][:,2].detach().cpu().numpy() , color='g', marker='o', alpha=0.5, s=1)#0.05
    
        ax.scatter(points2[index][:,0].detach().cpu().numpy(), 
                    points2[index][:,1].detach().cpu().numpy(),
                    points2[index][:,2].detach().cpu().numpy() , color='g', marker='o', alpha=0.5, s=1)
        
        ax.scatter(points1[index][coll_index1[index],0].detach().cpu().numpy(), 
                    points1[index][coll_index1[index],1].detach().cpu().numpy(),
                    points1[index][coll_index1[index],2].detach().cpu().numpy() , color='r', marker='o', alpha=1, s=2)
        ax.scatter(points2[index][coll_index2[index],0].detach().cpu().numpy(), 
                    points2[index][coll_index2[index],1].detach().cpu().numpy(),
                    points2[index][coll_index2[index],2].detach().cpu().numpy() , color='r', marker='o', alpha=1, s=2)

        for i, chain in enumerate(t2m_kinematic_chain):
            ax.plot3D(motion1[index, chain, 0].detach().cpu().numpy(), motion1[index, chain, 1].detach().cpu().numpy(), motion1[index, chain, 2].detach().cpu().numpy(), linewidth=1.0,
                        color='black')
            ax.plot3D(motion2[index, chain, 0].detach().cpu().numpy(), motion2[index, chain, 1].detach().cpu().numpy(), motion2[index, chain, 2].detach().cpu().numpy(), linewidth=1.0,
                        color='black')


  

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=range(frame_number), interval=1000 / fps, repeat=False)

    save_path = 'tmp.mp4'
    ani.save(save_path, fps=fps)
    plt.close()





def sample_points_on_cricle_surface_c(R, joints_line, device):
    #r [B,T,19,1]
    #joints_line [B,T,19,2,3]
    #return B,T,19*100,3
    # num_points_per_side =10
    num_points_c = 10
    num_points_line = 10 #10 5
    num_points = num_points_c*num_points_line
    #
    B,T = joints_line.shape[:2]
    P1 =  joints_line[:,:,:,0]#[B，T，19,3]
    P2 =  joints_line[:,:,:,1]
    v = P1 - P2
    u = v / torch.linalg.norm(v)  #[B，T，19，3]
    random_vector = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

    #  normalize the vectors a and b
    a = torch.cross(u,random_vector)
    b = torch.cross(u, a)
    eps = 1e-10
    a = a /(torch.linalg.norm(a, dim=-1, keepdim=True) + eps)
    b = b /(torch.linalg.norm(b, dim=-1, keepdim=True) + eps)

    
    # generate uniform sampling points on the circle
    theta = torch.linspace(0, 2 * torch.pi, num_points_c).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
    circle_points_P1 = P1.unsqueeze(-1) + R.unsqueeze(1).repeat(1,T,1).unsqueeze(-1).unsqueeze(-1)  * (torch.cos(theta) * a.unsqueeze(-1) + torch.sin(theta) * b.unsqueeze(-1))
    circle_points_P2 = P2.unsqueeze(-1) + R.unsqueeze(1).repeat(1,T,1).unsqueeze(-1).unsqueeze(-1)  * (torch.cos(theta) * a.unsqueeze(-1) + torch.sin(theta) * b.unsqueeze(-1))#[B,T,19,3,10]

    # generate uniform sampling points between P1 and P2
    n_samples = num_points_line
    # define a function to calculate uniform sampling points on the line
    def sample_line(x, y, n_samples):
        #x [B,T,N,3,1]
        t_values = torch.linspace(0, 1, n_samples).to(device)  # uniform sampling from 0 to 1
        t_values = t_values.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
        return (1 - t_values) * x + t_values * y  # linear interpolation formula

    # generate uniform sampling points between each pair of points
    line_samples = [sample_line(circle_points_P1[:, :,:,:,i].unsqueeze(-1), circle_points_P2[:,:,:,:,i].unsqueeze(-1), n_samples) for i in range(circle_points_P1.shape[4])]#[B,T,19,3,400]
    # merge all sampling points
    line_samples = torch.stack(line_samples, dim=-1).reshape(B,T,19,3,num_points)
    line_samples = line_samples.permute(0, 1, 2, 4, 3).reshape(B,T,19*num_points,3)
    
    #calculate op
    line_samples_op_0 = line_samples.reshape(B,T,19,num_points_line*2,int(num_points_c/2),3)
    line_samples_op_l = torch.roll(line_samples_op_0, shifts=-1, dims=3)
    line_samples_op_r = torch.roll(line_samples_op_0, shifts=1, dims=3)
    line_samples_op = line_samples_op_l
    line_samples_op[:,:,:,1::2] = line_samples_op_r[:,:,:,1::2]
    line_samples_op = line_samples_op_l.reshape(B,T,19*num_points,3)
    
    return line_samples, line_samples_op

def is_point_inside_cylinder(points, joints, radius):
    """
    This function is used to check if a point is inside a cylinder.
    :param points: points coordinates of shape (B, T, N, 3)
    :param joints: coordinates of the bottom and top centers of the cylinder (B, T, 19, 2, 3)
    :param radius: r (B, 19)
    :return: boolean tensor of shape (B, T, N, 19), indicating whether each point is inside any of the 19 cylinders
    """
    # extract the bottom and top centers of the cylinder
    B,T,N = points.shape[:3]
    M = joints.shape[2]
    C1 = joints[:, :, :, 0, :]  # (B, T, 19, 3)
    C2 = joints[:, :, :, 1, :]  # (B, T, 19, 3)

    # calculate the axis vector and normalize it
    v = C2 - C1  # (B, T, 19, 3)
    w = points.unsqueeze(3) - C1.unsqueeze(2)  # (B, T,  N,19, 3)
    # calculate the t value projected onto the axis
    t = torch.sum(w * v.unsqueeze(2), dim=-1) / torch.sum(v * v, dim=-1).unsqueeze(2)  # (B, T, N, 19)

    # check if t is out of range, if so, the point is not inside the cylinder
    inside_height = torch.logical_and(t >=0, t <=1)  # (B, T, N, 19)

    # limit t to be between 0 and 1
    t = torch.clamp(t, 0, 1) # (B, T, N, 19)

    # calculate the nearest point
    nearest_point = C1.unsqueeze(2) + t.unsqueeze(-1) * v.unsqueeze(2)# # (B, T, N, 19,3)

    # calculate the shortest distance
    distance = torch.linalg.norm(points.unsqueeze(3) - nearest_point, dim=-1)#[B,T,N,M]

    # handle the radius dimension
    radius = radius.unsqueeze(1).unsqueeze(1).repeat(1,T,N,1)  # (B, T, N, M,)

    # check if the point is inside the radius
    inside_radius = distance <= radius
    mask = torch.logical_and(inside_height, inside_radius)  # (B, T, N, 19)# only if the point is inside the height and radius, the point is inside the cylinder
    
    return  torch.any(mask, dim=-1)  # (B, T, N)

def compute_coll_loss_end_batch_c(prediction, mask, param1,param2):
    '''
    prediction : [batch, T, 2, 262]
    mask: [batch, T, 2, 1]
    param1: 4 
        'width' [batch, 19]
        'height' [batch, 19]
    '''
    
    device = prediction.device
    batch, T = prediction.shape[:2]
    mse_loss = nn.MSELoss()
    loss = torch.tensor(0.0).to(device)

    motion1 = prediction[:,:,0,:66].reshape(batch, T, 22, 3)
    motion2 = prediction[:,:,1,:66].reshape(batch, T, 22, 3)
    mask1 = mask[:,:,0] #[batch, T, 1]
    mask2 = mask[:,:,1]
    r_p1 = param1['r'].reshape(batch,19) 
    r_p2 = param2['r'].reshape(batch,19) 


    if torch.isnan(r_p1).any() :
        print("r_p1 is NaN. Stopping Training.")
    if torch.isnan(r_p2).any() :
        print("r_p2 is NaN. Stopping Training.")

    # 采样所有的点
    body_points1, body_points1_op = sample_points_on_cricle_surface_c(r_p1,motion1[:,:,box_smplx], device)#[batch, T, N, 3]  # N = 19*4*8*8 
    body_points2, body_points2_op = sample_points_on_cricle_surface_c(r_p2,motion2[:,:,box_smplx], device)#[batch, T, N, 3]

    collision_mask1 = is_point_inside_cylinder(body_points1,  motion2[:,:,box_smplx], r_p2) #[batch, T, N]   
    collision_mask2 = is_point_inside_cylinder(body_points2, motion1[:,:,box_smplx],r_p1)

    
    ##extract the valid frames of the collision points
    mask_1 = collision_mask1 & mask1.bool()#[batch, T, N]
    mask_2 = collision_mask2 & mask2.bool()

    # show the video of the collision points
    # for i in range(8):
    #     if mask_1[i].any() == True:
    #         batch_index = i
    #         show_video_c(motion1[batch_index], motion2[batch_index], body_points1[batch_index], body_points2[batch_index],collision_mask1[batch_index],collision_mask2[batch_index])
    #         import pdb
    #         pdb.set_trace()
    #         break

    eps = 1e-8
    if mask_1.any() == True:#there are collision points
        coll_1 = body_points1[mask_1]
        target_1 = body_points1_op[mask_1].detach().requires_grad_(False)
        direction = target_1 - coll_1  # [N, 3]
        direction_norm = torch.norm(direction, dim=-1, keepdim=True)  # avoid zero division
        direction_unit = direction / (direction_norm + eps)# normalize the direction vector [N, 3]
        x_target = (coll_1 + direction_unit).detach()  # [N, 3]
        loss += mse_loss(coll_1, x_target)/ (mask1.sum())

        
    if mask_2.any() == True:#there are collision points
        coll_2 = body_points2[mask_2]
        target_2 = body_points2_op[mask_2].detach().requires_grad_(False)
        direction2 = target_2 - coll_2  # [N, 3]
        direction_norm2 = torch.norm(direction2, dim=-1, keepdim=True)   # avoid zero division
        direction_unit2 = direction2 / (direction_norm2 + eps) # normalize the direction vector [N, 3]
        x_target2 = (coll_2 + direction_unit2).detach()  # [N, 3]
        loss += mse_loss(coll_2, x_target2)/ ( mask2.sum())

    return loss 