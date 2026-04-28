import torch
import torch.nn as nn
import torch.nn.functional as F
from .param import *

# 定义长方体的六个面，通过顶点索引来定义
plane_indices = torch.tensor([  # 将面索引定义为 tensor
    [0, 1, 2],  # 下面
    [4, 5, 6],  # 上面
    [0, 1, 4],  # 左面
    [3, 2, 7],  # 右面
    [1, 2, 5],  # 前面
    [0, 3, 4]   # 后面
    ])  # [6, 3]

# 定义每个长方体的四个面 (省略上下两面)
face_indices = torch.tensor([
    [0, 1, 5, 4],  # 左面
    [2, 3, 7, 6],  # 右面
    [0, 3, 7, 4],  # 后面
    [1, 2, 6, 5]   # 前面
])

face_indices_op = torch.tensor([
        [2, 3, 7, 6],  # 右面
        [0, 1, 5, 4],  # 左面
        [1, 2, 6, 5],   # 前面
        [0, 3, 7, 4]  # 后面
])

def calculate_cuboid_vertices(points, plane_points, width, height, device):
    '''
    points: [batch, T, 19, 2, 3]  骨架点坐标
    plane_points: [b,t,1*3]  方向
    width: [batch, 19]   
    height: [batch, 19]
    
    return: [batch, T, 19, 8, 3]
    '''
    B, T = points.shape[:2]
    width = width.to(device)
    height = height.to(device)
    num_box = points.shape[2]
    eps = 1e-8

    # 计算中心轴和长度
    p1 = points[:,:,:,0]
    p2 = points[:,:,:,1]

    # 1. 高的方向 
    direction = p2 - p1 #[B, T, 19, 3]   
    length = torch.linalg.norm(direction, dim=-1)#长度 [B,T,19]
    direction = direction / (length.unsqueeze(-1).expand(-1,-1,-1,3) + eps )# 单位化方向向量

    # 2. 计算plane_points 法向量 (该法向量与 plane_points 所在的平面垂直)
    normal_vector = torch.cross(plane_points[:,:,:,1] - plane_points[:,:,:,0], plane_points[:,:,:,2] - plane_points[:,:,:,0]) #[B,T,3]
    normal_vector = normal_vector / (torch.linalg.norm(normal_vector, dim=-1).unsqueeze(-1).expand(-1,-1, 3)+ eps)  # 单位化法向量 #[B,T,3]

    # 3. 宽的方向    选择与该平面法向量和高方向均正交的方向作为宽的方向
    normal_vector = normal_vector.unsqueeze(2).expand(-1, -1, num_box, -1) 
    width_vector = torch.cross(direction, normal_vector) #[B, T, 19, 3]
    width_vector = width_vector / (torch.linalg.norm(width_vector , dim=-1).unsqueeze(-1).expand(-1,-1,-1, 3)+ eps) #[B, T, 19, 3]
    
    # 4. 长的方向
    height_vector = torch.cross(direction, width_vector)
    height_vector = height_vector / (torch.linalg.norm(height_vector, dim=-1).unsqueeze(-1).expand(-1,-1,-1, 3)+ eps)  #[B, T, 19, 3]
    
    # 5. 长方体中心点、8个顶点的偏移量
    center = (p1 + p2) / 2 #[B, T, 19, 3]
    half_length = length / 2 #[B,t,19]
    half_width = width.unsqueeze(1).expand(-1,T,-1) / 2#[B,t,19]
    half_height = height.unsqueeze(1).expand(-1,T,-1) / 2#[B,t,19]
    half_dim = torch.stack([half_width, half_height, half_length], dim=-1).unsqueeze(-2).to(device) #[B,T,19,1,3]
    offsets = torch.tensor([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
    ], device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0) * half_dim #[B, T, 19, 8, 3]

    # 旋转和移动长方体
    rotation_matrix = torch.stack((width_vector, height_vector, direction), dim=-1) #[b,t,19,3, 3]

    vertices = center.unsqueeze(-2) + torch.einsum('...ij,...jk->...ik', offsets, rotation_matrix.transpose(-2, -1))#[B, T, 19, 8, 3]

    return vertices#[B, T, 19, 8, 3]


def sample_points_on_boxes_surface(box_vertices, num_points_per_side, device):
    """
    box_vertices:         [batch, T, N, 8, 3]
    num_points_per_side:  每个边采样的点数 8

    return: [batch, T, 19 * 4 * num_points_per_side^2, 3]
    """
    batch, frame = box_vertices.shape[:2]

    # 扩展 face_indices 以适应所有 box
    face_vertices = box_vertices[:, :, :, face_indices.to(device)]  # [batch, frame, 19, num_face, 4, 3)  19个长方体，4个面的4 个顶点
    
    # 生成线性插值的网格点
    grid_s, grid_t = torch.meshgrid(
        torch.linspace(0, 1, num_points_per_side, device=device),
        torch.linspace(0, 1, num_points_per_side, device=device)
    )
    grid_s, grid_t = grid_s.flatten(), grid_t.flatten()  # 将网格点展开为 1D 向量，便于插值
    
    # 计算插值点
    v0 = face_vertices[..., 0, :]  # [batch, frame, 19, num_face, 3]  4个面的 最左上角的 顶点坐标
    v1 = face_vertices[..., 1, :]  # [batch, frame, 19, num_face, 3]
    v2 = face_vertices[..., 2, :]  # [batch, frame, 19, num_face, 3]
    v3 = face_vertices[..., 3, :]  # [batch, frame, 19, num_face, 3]

    # 将 grid_s 和 grid_t 扩展为 2D 网格形式
    grid_s_exp = grid_s.unsqueeze(1).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 1, 100, 1]
    grid_t_exp = grid_t.unsqueeze(1).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 1, 100, 1]

    # 计算插值点：基于每个面的四个顶点
    points = (1 - grid_s_exp) * ((1 - grid_t_exp) * v0.unsqueeze(-2)+ \
                                 grid_t_exp * v1.unsqueeze(-2)) + \
             grid_s_exp * ((1 - grid_t_exp) * v3.unsqueeze(-2) + \
                           grid_t_exp * v2.unsqueeze(-2))    

    face_points_all = points.view(batch, frame, -1, 3)# [batch, frame, 19 * 4 * 8*8, 3]

    #-------op-------
    face_vertices_op = box_vertices[:, :, :, face_indices_op.to(device)]
    # 计算插值点
    v0_op = face_vertices_op[..., 0, :]  # [batch, frame, num_cuboids, num_faces, 3]
    v1_op = face_vertices_op[..., 1, :]  # [batch, frame, num_cuboids, num_faces, 3]
    v2_op = face_vertices_op[..., 2, :]  # [batch, frame, num_cuboids, num_faces, 3]
    v3_op = face_vertices_op[..., 3, :]  # [batch, frame, num_cuboids, num_faces, 3]
    points_op = (1 - grid_s_exp) * ((1 - grid_t_exp) * v0_op.unsqueeze(-2)+ grid_t_exp * v1_op.unsqueeze(-2)) + \
             grid_s_exp * ((1 - grid_t_exp) * v3_op.unsqueeze(-2) + grid_t_exp * v2_op.unsqueeze(-2))
    face_points_all_op = points_op.view(batch, frame, -1, 3)

    return   face_points_all , face_points_all_op# [batch, frame, 19 * 4 * num_points_per_face^2, 3]


def find_penetration_points_(points, cuboids_vertices):
    """
    points:[batch, T, N, 3]  
    cuboids_vertices:  [batch,T, 19, 8, 3] 
    
    return  [batch, T, N]
    """
    planes = cuboids_vertices[:, :, :, plane_indices.to(cuboids_vertices.device)]  # [batch, T, B, 6, 3, 3]
    # Compute the two edge vectors of each face
    v1 = planes[:,:,:, :, 1] - planes[:,:,:, :, 0]  # (batch, T, B, 6, 3)
    v2 = planes[:,:,:, :, 2] - planes[:,:,:, :, 0]  # (batch, T, B, 6, 3)
    # Compute face normals
    face_normal = torch.cross(v1, v2, dim=-1)  # (batch, T, B, 6, 3)
    # Take one vertex from each face
    face_point = planes[:,:, :, :, 0]  # (batch, T, B, 6, 3)
    # Compute the vector from each face point to each query point
    points_expanded = points.unsqueeze(3).unsqueeze(3)  # (batch, T, N, 1, 1, 3)
    face_point_expanded = face_point.unsqueeze(2)  # (batch, T, 1, 19, 6, 3)
    points_normal = points_expanded - face_point_expanded  # (batch, T, N, 19, 6, 3)
    # Compute the dot product of each point with each face normal
    # i.e. dot product of N points against 6 faces of B cuboids
    dot_products = torch.einsum('xynbmc,xybmc->xynbm', points_normal, face_normal)  # (batch, T, N, 19, 6)

    # conditions: collision status of each point against each cuboid, each element is a boolean
    conditions = (dot_products[:,:,:, :, 0] * dot_products[:,:,:, :, 1] <= 0) & \
                 (dot_products[:,:,:, :, 2] * dot_products[:,:,:, :, 3] <= 0) & \
                 (dot_products[:,:,:, :, 4] * dot_products[:,:,:, :, 5] <= 0)  # [batch, T, N, 19]

    penetration_indices = torch.any(conditions, dim=3)  # get indices satisfying the condition [batch, T, N]

    return penetration_indices

def convert_to_aligned_coords_batch(points, cuboid_vertices, mask):
    """
    cuboid_vertices [batch, T, 19, 8, 3] - 8 vertex coordinates of each cuboid
    points [batch, T, N, 3] - coordinates of N points
    mask: [batch, T, 1]
    Output:
        aligned_vertices [batch, T, 19, 8, 3] - cuboid vertices in the transformed coordinate system
        aligned_points [batch, T, 19, N, 3] - points in the transformed coordinate system
    """
    eps = 1e-8
    cuboid_center = cuboid_vertices.mean(dim=3, keepdim=True)    # Compute cuboid centers [batch, T, 19, 1, 3]
    
    # Compute the directions of the x, y, z axes
    axis_x = cuboid_vertices[:, :, :, 1, :] - cuboid_vertices[:, :, :, 0, :]  # x-axis direction
    axis_y = cuboid_vertices[:, :, :, 3, :] - cuboid_vertices[:, :, :, 0, :]  # y-axis direction
    axis_z = cuboid_vertices[:, :, :, 4, :] - cuboid_vertices[:, :, :, 0, :]  # z-axis direction

    #norm [batch, T, 19, 3] 
    axis_x = axis_x / (torch.norm(axis_x, dim=-1, keepdim=True) + eps)
    axis_y = axis_y / (torch.norm(axis_y, dim=-1, keepdim=True) + eps)
    axis_z = axis_z / (torch.norm(axis_z, dim=-1, keepdim=True) + eps)

    # Construct the rotation matrix [batch, T, 19, 3, 3]
    rotation_matrix = torch.stack([axis_x, axis_y, axis_z], dim=-1)

    # Translate the cuboid vertices
    centered_vertices = cuboid_vertices - cuboid_center  # [batch, T, 19, 8, 3]

    # Rotate the cuboid vertices
    aligned_vertices = torch.einsum('abijc,abick->abijk', centered_vertices, rotation_matrix)  # [batch, T, 19, 8, 3]

    # Apply translation and rotation to multiple points
    centered_points = points.unsqueeze(2) - cuboid_center  # [batch, T, 19, N, 3]
    aligned_points = torch.einsum('abijc, abick->abijk', centered_points, rotation_matrix)  # [batch, T, 19, N, 3]

    return aligned_points, aligned_vertices

def convert_to_aligned_coords(points, cuboid_vertices):
    """
    Input:  cuboid_vertices [batch*T, 19, 8, 3] - 8 vertex coordinates of each cuboid
            points [batch*T, N, 3] - coordinates of N points
    Output:
            aligned_vertices [batch*T, 19, 8, 3] - cuboid vertices in the transformed coordinate system
            aligned_points [batch*T, 19, N, 3] - points in the transformed coordinate system
    """
    # Compute cuboid centers [batch*T, 19, 1, 3]
    batch, T, N = points.shape[:3]
    cuboid_vertices = cuboid_vertices.reshape(batch*T, 19, 8, 3)
    points = points.reshape(batch*T, N, 3)
    cuboid_center = cuboid_vertices.mean(dim=2, keepdim=True)
    
    # Compute the directions of the x, y, z axes
    axis_x = cuboid_vertices[:, :, 1, :] - cuboid_vertices[:, :, 0, :]  # x-axis direction
    axis_y = cuboid_vertices[:, :, 3, :] - cuboid_vertices[:, :, 0, :]  # y-axis direction
    axis_z = cuboid_vertices[:, :, 4, :] - cuboid_vertices[:, :, 0, :]  # z-axis direction

    # Normalize the axes [batch*T, 19, 3]
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = axis_z / torch.norm(axis_z, dim=-1, keepdim=True)

    # Construct the rotation matrix [batch*T, 19, 3, 3]
    rotation_matrix = torch.stack([axis_x, axis_y, axis_z], dim=-1)

    # Apply translation and rotation to the cuboid vertices
    centered_vertices = cuboid_vertices - cuboid_center  # [batch*T, 19, 8, 3]

    aligned_vertices = torch.einsum('bijc,bick->bijk', centered_vertices, rotation_matrix)  # [batch*T, 19, 8, 3]

    # Apply translation and rotation to multiple points
    centered_points = points.unsqueeze(1) - cuboid_center  # [batch*T, 19, N, 3]
    aligned_points = torch.einsum('bijc,bick->bijk', centered_points, rotation_matrix)  # [batch*T, 19, N, 3]

    return  aligned_points.reshape(batch, T, 19, N, 3), aligned_vertices.reshape(batch, T, 19, 8, 3)

def find_penetration_points_ro(points, cuboids_vertices, mask):
    """
    points: [batch, T, N, 3]  coordinates of multiple points
    cuboids_vertices: 8 vertices of multiple cuboids [batch, T, 19, 8, 3]
    
    return  [batch, T, N]
    """

    aligned_points, aligned_box = convert_to_aligned_coords_batch(points, cuboids_vertices, mask) #[batch, T, 19, N, 3]   [batch, T, 19, 8, 3]  #10ms
    
    # Compute the minimum and maximum points of the cuboid
    aligned_box_min = aligned_box.min(dim=3).values  # [batch, T, 19, 3]
    aligned_box_max = aligned_box.max(dim=3).values  # [batch, T, 19, 3]

    # Check whether the point is within the cuboid bounds
    inside = (aligned_points >= aligned_box_min.unsqueeze(3)) & (aligned_points <= aligned_box_max.unsqueeze(3))

    collision = inside.all(dim=-1).any(dim=2)  # [batch, T, N]
    return collision


def compute_coll_loss_end_batch(prediction, mask, num_points, param1, param2):
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
    width_p1 = param1['width'].reshape(batch,19)
    width_p2 = param2['width'].reshape(batch,19)
    height_p1 = param1['height'].reshape(batch,19)
    height_p2 = param2['height'].reshape(batch,19)


    if torch.isnan(width_p1).any() :
        print("width_p1 is NaN. Stopping Training.")
    if torch.isnan(width_p2).any() :
        print("width_p2 is NaN. Stopping Training.")
    if torch.isnan(height_p1).any() :
        print("height_p1 is NaN. Stopping Training.")
    if torch.isnan(height_p2).any() :
        print("height_p2 is NaN. Stopping Training.")


    # get the vertices of the boxes
    boxes_list1 = calculate_cuboid_vertices(motion1[:, :, box], motion1[:,:, body_direction], width_p1, height_p1, device)#[batch, T, 19,8,3]
    boxes_list2 = calculate_cuboid_vertices(motion2[:, :, box], motion2[:,:, body_direction], width_p2, height_p2, device)#[batch, T, 19,8,3]

    # sample all the points on the boxes
    body_points1, body_points1_op = sample_points_on_boxes_surface(boxes_list1, num_points, device)#[batch, T, N, 3]  # N = 19*4*8*8 
    body_points2, body_points2_op = sample_points_on_boxes_surface(boxes_list2, num_points, device)#[batch, T, N, 3]

    ## find_penetration_points_ and find_penetration_points_ro are functionally equivalent, differing only in computational speed
    # collision_mask1_ = find_penetration_points_(body_points1, boxes_list2) #[batch, T, N]   
    # collision_mask2_ = find_penetration_points_(body_points2, boxes_list1)

    collision_mask1 = find_penetration_points_ro(body_points1, boxes_list2, mask1) #[batch, T, N]   
    collision_mask2 = find_penetration_points_ro(body_points2, boxes_list1, mask2)

    
    ## get the valid collision points
    mask_1 = collision_mask1 & mask1.bool()#[batch, T, N]
    mask_2 = collision_mask2 & mask2.bool()

    eps = 1e-8
    if mask_1.any() == True:#there are collision points
        coll_1 = body_points1[mask_1]
        target_1 = body_points1_op[mask_1].detach().requires_grad_(False)
        direction = target_1 - coll_1  # [N, 3]
        direction_norm = torch.norm(direction, dim=-1, keepdim=True)  #
        direction_unit = direction / (direction_norm + eps)#  [N, 3]
        x_target = (coll_1 + direction_unit).detach()  # [N, 3]
        loss += mse_loss(coll_1, x_target)/ (mask1.sum())

        
    if mask_2.any() == True:
        coll_2 = body_points2[mask_2]
        target_2 = body_points2_op[mask_2].detach().requires_grad_(False)
        direction2 = target_2 - coll_2  # [N, 3]
        direction_norm2 = torch.norm(direction2, dim=-1, keepdim=True)  
        direction_unit2 = direction2 / (direction_norm2 + eps)
        x_target2 = (coll_2 + direction_unit2).detach()  # [N, 3]
        loss += mse_loss(coll_2, x_target2)/ ( mask2.sum())

    return loss 





def points_coll_loss(points1, points2, mask, device):
    '''
    points : [B,T,11*16*3]
    '''
    mse_loss = nn.MSELoss()
    loss = torch.tensor(0.0).to(device) 
    B,T = points1.shape[:2]
    points1_box = points1.reshape(B,T,11,16,3)[:,:,:,:8] #[B,T,11,8,3]  take the first 8 points representing the box vertices
    points2_box = points2.reshape(B,T,11,16,3)[:,:,:,:8]

    body_points_1, body_points_1_op = sample_points_on_boxes_surface(points1_box, 5, device)#[batch, T, 11 * 4 * num_points_per_face^2, 3]
    body_points_2, body_points_2_op = sample_points_on_boxes_surface(points2_box, 5, device)

    points1_center = points1.reshape(B,T,11,16,3)[:,:,:,8:].reshape(B,T,11*8,3)
    points1_center_op = points1_center.clone()
    points1_center_op[:,:,0:2] = points1_center[:,:,2:4]
    points1_center_op[:,:,2:4] = points1_center[:,:,0:2]
    points1_center_op[:,:,4:6] = points1_center[:,:,6:8]
    points1_center_op[:,:,6:8] = points1_center[:,:,4:6]
    body_points1 = torch.cat((body_points_1, points1_center), dim=2)
    body_points1_op = torch.cat((body_points_1_op, points1_center_op), dim=2)

    points2_center = points2.reshape(B,T,11,16,3)[:,:,:,8:].reshape(B,T,11*8,3)
    points2_center_op = points2_center.clone()
    points2_center_op[:,:,0:2] = points2_center[:,:,2:4]
    points2_center_op[:,:,2:4] = points2_center[:,:,0:2]
    points2_center_op[:,:,4:6] = points2_center[:,:,6:8]
    points2_center_op[:,:,6:8] = points2_center[:,:,4:6]
    body_points2 = torch.cat((body_points_2, points2_center), dim=2)
    body_points2_op = torch.cat((body_points_2_op, points2_center_op), dim=2)


    collision_mask1 = find_penetration_points_ro(body_points1, points2_box, mask) #[batch, T, N]   
    collision_mask2 = find_penetration_points_ro(body_points2, points1_box, mask)

    ##取出有效帧的碰撞点
    mask_1 = collision_mask1 & mask.bool()#[batch, T, N]
    mask_2 = collision_mask2 & mask.bool()

    eps = 1e-8
    if mask_1.any() == True:#有碰撞点
        coll_1 = body_points1[mask_1]
        target_1 = body_points1_op[mask_1].detach().requires_grad_(False)
        direction = target_1 - coll_1  # [N, 3]
        direction_norm = torch.norm(direction, dim=-1, keepdim=True)  # avoid zero division
        direction_unit = direction / (direction_norm + eps)# normalize the direction vector [N, 3]
        x_target = (coll_1 + direction_unit).detach()  # [N, 3]
        loss += mse_loss(coll_1, x_target)/ (mask_1.sum())
        
    if mask_2.any() == True:#有碰撞点
        coll_2 = body_points2[mask_2]
        target_2 = body_points2_op[mask_2].detach().requires_grad_(False)
        x_target2 = target_2.detach()
        direction2 = target_2 - coll_2  
        direction_norm2 = torch.norm(direction2, dim=-1, keepdim=True)   # avoid zero division
        direction_unit2 = direction2 / (direction_norm2 + eps) # normalize the direction vector [N, 3]
        x_target2 = (coll_2 + direction_unit2).detach() 
        loss += mse_loss(coll_2, x_target2)/ ( mask_2.sum())

    return loss 
