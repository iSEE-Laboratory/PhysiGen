import numpy as np
import torch.nn as nn
import torch

OPTITRACK_SKEL=[
    'Hips',
    'RightUpLeg','RightLeg','RightFoot','RightToeBase',# 'RightToeBase_Nub',
    'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase',# 'LeftToeBase_Nub',
    'Spine','Spine_1',
    'RightShoulder','RightArm','RightForeArm','RightHand',
        'RightHandPinky1','RightHandPinky2','RightHandPinky3','RightHandPinky3_Nub',
        'RightHandRing1','RightHandRing2','RightHandRing3','RightHandRing3_Nub',
        'RightHandMiddle1','RightHandMiddle2','RightHandMiddle3','RightHandMiddle3_Nub',
        'RightHandIndex1','RightHandIndex2','RightHandIndex3','RightHandIndex3_Nub',
        'RightHandThumb1','RightHandThumb2','RightHandThumb3','RightHandThumb3_Nub',
    'LeftShoulder','LeftArm','LeftForeArm','LeftHand',
        'LeftHandPinky1','LeftHandPinky2','LeftHandPinky3','LeftHandPinky3_Nub',
        'LeftHandRing1','LeftHandRing2','LeftHandRing3','LeftHandRing3_Nub',
        'LeftHandMiddle1','LeftHandMiddle2','LeftHandMiddle3','LeftHandMiddle3_Nub',
        'LeftHandIndex1','LeftHandIndex2','LeftHandIndex3','LeftHandIndex3_Nub',
        'LeftHandThumb1','LeftHandThumb2','LeftHandThumb3','LeftHandThumb3_Nub',
    'Neck','Head',# 'Head_Nub'
] # 64 , 61 used,xxx_Nub is ignored hand_Nub can not be ignored

OPTITRACK_BODY_smplx=np.concatenate(
    [range(0, 25)]
)
OPTITRACK_HAND_smplx=np.concatenate(
    [range(25,55)]
)

OPTITRACK_BODY_smplh=np.concatenate(
    [range(0, 22)]
)
OPTITRACK_HAND_smplh=np.concatenate(
    [range(22,52)]
)
# OPTITRACK_HAND=np.concatenate(
#     [range(22,52)]
# )

INTERX_TO_SMPLX=np.array(range(55))

INTERX_TO_SMPLH=np.array(range(52))

class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)
        
if __name__=='__main__':
    print(INTERX_TO_SMPLX)