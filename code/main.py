"""
This code is based on  https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
"""
from models.gaussian_diffusion import GaussianDiffusion
from models.motion_normalizer import MotionNormalizerTorch
from models.geometric_loss import GeometricLoss
from models.inter_loss import InterLoss
from models.losses import InterLoss, GeometricLoss
import torch
import numpy as np
#---------------------------------------------------------------------------------------------
from collision.coll_loss import  compute_coll_loss_end_batch
from collision.coll_loss_c import  compute_coll_loss_end_batch_c
from collision.coll_loss_mesh import  compute_coll_loss_mesh
from visualize.utils.mapping import (INTERX_TO_SMPLX, INTERX_TO_SMPLH, JointMapper)
from smplx import SMPLXLayer, SMPLHLayer

body_models = SMPLHLayer(model_path='./smpl_models/smplh',num_betas=10,gender='neutral',
                        joint_mapper=JointMapper(INTERX_TO_SMPLH),flat_hand_mean=False,ext='npz') 

class MotionDiffusion(GaussianDiffusion):

    def __init__(self, use_timesteps, motion_rep, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.motion_rep = motion_rep
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        # print(self.timestep_map)
        kwargs["betas"] = np.array(new_betas)

        self.normalizer = MotionNormalizerTorch()

        super().__init__(**kwargs)

    def p_mean_variance(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, mask, t_bar, cond_mask, param1, param2, *args, **kwargs):
        def check_nan(x, name="Variable"):
            if torch.isnan(x).any():
                print(f"NaN detected in {name}")

        target0 = kwargs["x_start"]
        B, T = target0.shape[:-1]
        target0 = target0.reshape(B, T, 2, -1)
        mask = mask.reshape(B, T, -1, 1)
        target = self.normalizer.forward(target0)
        kwargs["x_start"] = target.reshape(B, T, -1)

        items = super().training_losses(self._wrap_model(model), *args, **kwargs)

        prediction = items["pred"].reshape(B, T, 2, -1)
        target = items["target"].reshape(B, T, 2, -1)
        check_nan(prediction, "prediction nan")
        check_nan(target, "target nan")

        timestep_mask = (kwargs["t"] <= t_bar).float()  # * cond_mask[0, 0]


        interloss_manager = InterLoss("l2", 22)
        interloss_manager.forward(prediction, target, mask, timestep_mask)

        loss_a_manager = GeometricLoss("l2", 22, "A")
        loss_a_manager.forward(prediction[...,0,:], target[...,0,:], mask[...,0,:], timestep_mask)

        loss_b_manager = GeometricLoss("l2", 22, "B")
        loss_b_manager.forward(prediction[...,1,:], target[...,1,:], mask[...,0,:], timestep_mask)

        losses = {}
        losses.update(loss_a_manager.losses)
        losses.update(loss_b_manager.losses)
        losses.update(interloss_manager.losses)
        losses["total"] = loss_a_manager.losses["A"] + loss_b_manager.losses["B"]  + interloss_manager.losses["total"]  

        prediction_ =  self.normalizer.backward(prediction, global_rt=True)

        #-----------------------------------------main code--------------------------------------------------------
        num_points = 3  #choose the num of the points on the line of the bounding box for collision detection
        weight = 0.1 # choose weight for the collision loss
        points_sampled_on_mesh = 6890 # the num of the points on the mesh for collision detection
        use_cylinder = True
        use_mesh = False
        use_box = False

        if use_box:#if use the box for collision detection
            loss_coll_box = compute_coll_loss_end_batch(prediction_, mask, num_points, param1, param2) * weight
            loss_box = {"loss_coll_box":loss_coll_box}
            losses["total"] += loss_coll_box
            losses.update(loss_box)
        elif use_cylinder:#if use the cylinder for collision detection
            loss_coll_cylinder = compute_coll_loss_end_batch_c(prediction_, mask, param1, param2) * weight
            loss_cylinder = {"loss_coll_cylinder":loss_coll_cylinder}
            losses["total"] += loss_coll_cylinder
            losses.update(loss_cylinder)
        elif use_mesh:#if use the mesh for collision detection
            loss_sdf = compute_coll_loss_mesh(prediction_, mask, param1, param2, body_models, points_sampled_on_mesh) * weight
            sdf = {"sdf_loss":loss_sdf}
            losses["total"] += loss_sdf
            losses.update(sdf)
        else:
            raise ValueError("Invalid collision detection method")
        #-----------------------------------------main code--------------------------------------------------------

        return losses