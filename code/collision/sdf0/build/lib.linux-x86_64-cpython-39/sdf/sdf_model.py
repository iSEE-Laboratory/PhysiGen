import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# import sdf.csrc as _C
import importlib    
import sdf
# print('sdf path:', sdf.__file__)             # 应为 …/site-packages/sdf/__init__.py
_C = importlib.import_module("sdf.csrc")     # 不再报错
# print('C++/CUDA ext loaded ->', _C)

class SDFFunction(Function):
    """
    Definition of SDF function
    """

    @staticmethod
    def forward(ctx, phi, faces, vertices, points):
        return _C.sdf(phi, faces, vertices, points)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

class SDF(nn.Module):
    def __init__(self, grid_size=6890):
        super(SDF, self).__init__()
        self.grid_size = grid_size # 采点数量

    def forward(self, faces, vertices, points):
        #输入是网格面片faces和顶点vertices， 
        # 输出是SDF值phi， 大小为(batch_size, num_points)
        
        phi = torch.zeros(vertices.shape[0], points.shape[1], device=vertices.device)
        phi = phi.contiguous()
        faces = faces.contiguous()
        vertices = vertices.contiguous()
        points = points.contiguous()

        phi = SDFFunction.apply(phi, faces, vertices, points)
        return phi


