"""
Transformation utils
"""

import numpy as np
import math
import torch
import torch.nn.functional as F
from opencood.utils.common_utils import check_numpy_to_torch

def muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C):
    rotationA2B = np.array(rotationA2B).reshape(3, 3)
    rotationB2C = np.array(rotationB2C).reshape(3, 3)
    rotation = np.dot(rotationB2C, rotationA2B)
    translationA2B = np.array(translationA2B).reshape(3, 1)
    translationB2C = np.array(translationB2C).reshape(3, 1)
    translation = np.dot(rotationB2C, translationA2B) + translationB2C

    return rotation, translation

def tfm_to_pose(tfm: np.ndarray):
    """
    turn transformation matrix to [x, y, z, roll, yaw, pitch]
    we use radians format.
    tfm is pose in transformation format, and XYZ order, i.e. roll-pitch-yaw
    """
    # These formulas are designed from x_to_world, but equal to the one below.
    yaw = np.degrees(np.arctan2(tfm[1, 0], tfm[0, 0])) # clockwise in carla
    roll = np.degrees(np.arctan2(-tfm[2, 1], tfm[2, 2])) # but counter-clockwise in carla
    pitch = np.degrees(np.arctan2(tfm[2, 0], ((tfm[2, 1] ** 2 + tfm[2, 2] ** 2) ** 0.5)) ) # but counter-clockwise in carla

    # These formulas are designed for consistent axis orientation
    # yaw = np.degrees(np.arctan2(tfm[1,0], tfm[0,0])) # clockwise in carla
    # roll = np.degrees(np.arctan2(tfm[2,1], tfm[2,2])) # but counter-clockwise in carla
    # pitch = np.degrees(np.arctan2(-tfm[2,0], ((tfm[2,1]**2 + tfm[2,2]**2) ** 0.5)) ) # but counter-clockwise in carla

    # roll = - roll
    # pitch = - pitch

    x, y, z = tfm[:3,3]
    return [x, y, z, roll, yaw, pitch]

def align_features(features, ego_poses, _d=1.6, target_pos=-1):
    """
    Align the features from previous ego pose to current ego pose
    features shape b T c h w
    ego_poses shape b T 4 4  #t means temporal frames
    _d: discrete_ratio * downsample_rate  discrete_ratio=0.4, downsample_rate=4
    aligned_features b T c h w
    """
    assert target_pos in [-1, 0]

    B, T, C, H, W = features.shape
    # ### type 1 #B T T 4 4
    pairwise_t_matrix = get_pairwise_transformation_torch(
        ego_poses, max_cav=T)

    pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
    # [B, L, L, 2, 3]
    pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
    pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
    pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (_d * W) * 2
    pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (_d * H) * 2
    x_fuse = []
    for b in range(B):
        #     # number of valid agent
        #     # (N,N,4,4)
        #     # t_matrix[i, j]-> from i to j
        t_matrix = pairwise_t_matrix[b][:T, :T, :, :]
        node_features = features[b]
        neighbor_feature = warp_affine_simple(
            node_features, t_matrix[target_pos, :, :, :], (H, W))
        x_fuse.append(neighbor_feature)
    aligned_features = torch.stack(x_fuse)
    # aligned_features = torch.cat((aligned_features[:,:-1:,...],features[:,-1:,...]),dim=1)
    return aligned_features


def pose_to_tfm(pose):
    """ Transform batch of pose to tfm
    Args:
        pose: torch.Tensor or np.ndarray
            [N, 3], x, y, yaw, in degree
            [N, 6], x, y, z, roll, yaw, pitch, in degree

            roll and pitch follows carla coordinate
    Returns:
        tfm: torch.Tensor
            [N, 4, 4] 
    """

    pose_tensor, is_np = check_numpy_to_torch(pose)
    pose = pose_tensor

    if pose.shape[1] == 3:
        N = pose.shape[0]
        x = pose[:, 0]
        y = pose[:, 1]
        yaw = pose[:, 2]

        tfm = torch.eye(4, device=pose.device).view(1, 4, 4).repeat(N, 1, 1)
        tfm[:, 0, 0] = torch.cos(torch.deg2rad(yaw))
        tfm[:, 0, 1] = - torch.sin(torch.deg2rad(yaw))
        tfm[:, 1, 0] = torch.sin(torch.deg2rad(yaw))
        tfm[:, 1, 1] = torch.cos(torch.deg2rad(yaw))
        tfm[:, 0, 3] = x
        tfm[:, 1, 3] = y

    elif pose.shape[1] == 6:
        N = pose.shape[0]
        x = pose[:, 0]
        y = pose[:, 1]
        z = pose[:, 2]
        roll = pose[:, 3]
        yaw = pose[:, 4]
        pitch = pose[:, 5]

        c_y = torch.cos(torch.deg2rad(yaw))
        s_y = torch.sin(torch.deg2rad(yaw))
        c_r = torch.cos(torch.deg2rad(roll))
        s_r = torch.sin(torch.deg2rad(roll))
        c_p = torch.cos(torch.deg2rad(pitch))
        s_p = torch.sin(torch.deg2rad(pitch))

        tfm = torch.eye(4, device=pose.device).view(1, 4, 4).repeat(N, 1, 1)

        # translation matrix
        tfm[:, 0, 3] = x
        tfm[:, 1, 3] = y
        tfm[:, 2, 3] = z

        # rotation matrix
        tfm[:, 0, 0] = c_p * c_y
        tfm[:, 0, 1] = c_y * s_p * s_r - s_y * c_r
        tfm[:, 0, 2] = -c_y * s_p * c_r - s_y * s_r
        tfm[:, 1, 0] = s_y * c_p
        tfm[:, 1, 1] = s_y * s_p * s_r + c_y * c_r
        tfm[:, 1, 2] = -s_y * s_p * c_r + c_y * s_r
        tfm[:, 2, 0] = s_p
        tfm[:, 2, 1] = -c_p * s_r
        tfm[:, 2, 2] = c_p * c_r

    if is_np:
        tfm = tfm.numpy()

    return tfm


def warp_affine_simple(src, M, dsize,
                       mode='bilinear',
                       padding_mode='zeros',
                       align_corners=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)


def warp_affine(
        src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True):
    r"""
    Transform the src based on transformation matrix M.
    Args:
        src : torch.Tensor
            Input feature map with shape :math:`(B,C,H,W)`.
        M : torch.Tensor
            Transformation matrix with shape :math:`(B,2,3)`.
        dsize : tuple
            Tuple of output image H_out and W_out.
        mode : str
            Interpolation methods for F.grid_sample.
        padding_mode : str
            Padding methods for F.grid_sample.
        align_corners : boolean
            Parameter of F.affine_grid.

    Returns:
        Transformed features with shape :math:`(B,C,H,W)`.
    """

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3 = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm = normalize_homography(M_3x3, (H, W), dsize)

    # src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)

    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :],
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners)

    return F.grid_sample(src.half() if grid.dtype == torch.half else src,
                         grid, align_corners=align_corners, mode=mode,
                         padding_mode=padding_mode)


def get_pairwise_transformation_torch(lidar_poses_list, max_cav=3):
    """
    Get pair-wise transformation matrix accross different agents.
    Designed for batch data

    Parameters
    ----------
    lidar_poses : tensor, [B, L, 3], [B, L, 6] or [B, L, 4, 4]
        3 or 6 dof pose of lidar.

    max_cav : int
        The maximum number of cav, default 5

    Return
    ------
    pairwise_t_matrix : np.array
        The pairwise transformation matrix across each cav.
        shape: (B, L, L, 4, 4), L is the max cav number in a scene
        pairwise_t_matrix[i, j] is Tji, i_to_j
    """

    B = lidar_poses_list.shape[0]

    pairwise_t_matrix = torch.eye(4, device=lidar_poses_list.device).view(
        1, 1, 1, 4, 4).repeat(B, max_cav, max_cav, 1, 1)  # (B, L, L, 4, 4)
    # save all transformation matrix in a list in order first.
    for b in range(B):
        t_list = lidar_poses_list[b]  # Twx, [N_cav, 4, 4]
        if t_list.shape[-1] != 4:
            # [N_cav, 3 or 6] to Twx, [N_cav, 4, 4]
            t_list = pose_to_tfm(t_list)

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i != j:
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    t_matrix = torch.linalg.solve(
                        t_list[j], t_list[i])  # Tjw*Twi = Tji
                    pairwise_t_matrix[b][i, j] = t_matrix

    return pairwise_t_matrix


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list or np.ndarray
        The pose of x1 under world coordinates or
        transformation matrix x1->world
    x2 : list or np.ndarray
        The pose of x2 under world coordinates or
         transformation matrix x2->world

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    if isinstance(x1, list) and isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        x2_to_world = x_to_world(x2)
        world_to_x2 = np.linalg.inv(x2_to_world)
        transformation_matrix = np.dot(world_to_x2, x1_to_world)

    # object pose is list while lidar pose is transformation matrix
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_world = x_to_world(x1)
        world_to_x2 = x2
        transformation_matrix = np.dot(world_to_x2, x1_to_world)
    # both are numpy matrix
    else:
        world_to_x2 = np.linalg.inv(x2)
        transformation_matrix = np.dot(world_to_x2, x1)

    return transformation_matrix


def dist_two_pose(cav_pose, ego_pose):
    """
    Calculate the distance between agent by given there pose.
    """
    if isinstance(cav_pose, list):
        distance = \
            math.sqrt((cav_pose[0] -
                       ego_pose[0]) ** 2 +
                      (cav_pose[1] - ego_pose[1]) ** 2)
    else:
        distance = \
            math.sqrt((cav_pose[0, -1] -
                       ego_pose[0, -1]) ** 2 +
                      (cav_pose[1, -1] - ego_pose[1, -1]) ** 2)
    return distance


def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous


# old codes
# def align(self, x, ego_poses,):  # 效果等同于取第三维度的sttf，from v2vfuse
#     from opencood.utils.transformation_utils import get_pairwise_transformation_torch
#     from opencood.models.sub_modules.torch_transformation_utils import \
#         get_transformation_matrix, warp_affine, get_discretized_transformation_matrix
#     B, T, C, H, W = x.shape
#     # import ipdb;ipdb.set_trace()
#     pairwise_t_matrix = get_pairwise_transformation_torch(
#         ego_poses, max_cav=T)
#     # x: (B,C,H,W)
#     # record_len: (B)
#     # pairwise_t_matrix: (B,L,L,4,4)
#     # prior_encoding: (B,3)
#     B, L = pairwise_t_matrix.shape[:2]
#     # split x:[(L1, C, H, W), (L2, C, H, W)]
#     # split_x = self.regroup(x, record_len)
#     # (B,L,L,2,3)
#     pairwise_t_matrix = get_discretized_transformation_matrix(
#         pairwise_t_matrix.reshape(-1, L, 4,
#                                   4), self.fusion_net.encoder.discrete_ratio,
#         self.fusion_net.encoder.downsample_rate).reshape(B, L, L, 2, 3)
#     batch_node_features = x
#     # iteratively update the features for num_iteration times
#     out = []
#     for b in range(B):

#         # t_matrix[i, j]-> from i to j
#         N = pairwise_t_matrix.shape[2]
#         t_matrix = pairwise_t_matrix[b]
#         current_t_matrix = t_matrix[:, -1, :, :]
#         current_t_matrix = get_transformation_matrix(
#             current_t_matrix, (H, W))
#         neighbor_feature = warp_affine(batch_node_features[b],
#                                        current_t_matrix,
#                                        (H, W))
#         # (N,C,H,W)
#         ego_agent_feature = batch_node_features[b][-1]
#         # (N,2C,H,W)
#         import ipdb
#         ipdb.set_trace()
#         neighbor_feature = torch.cat(
#             [neighbor_feature[:-1,], ego_agent_feature[None]], dim=0)
#         # (N,C,H,W)

#         out.append(neighbor_feature)
#     # (B,C,H,W)
#     out = torch.stack(out, dim=0)
#     return out

# def align_features(self, features, ego_poses, target_pos=-1):
#     """
#     Align the features from previous ego pose to current ego pose
#     features shape b T c h w
#     ego_poses shape b T 4 4  #t means temporal frames
#     aligned_features b T c h w
#     """
#     assert target_pos in [-1, 0]
#     from opencood.utils.transformation_utils import warp_affine_simple, x1_to_x2, get_pairwise_transformation_torch

#     B, T, C, H, W = features.shape
#     # ### type 1 #B T T 4 4
#     pairwise_t_matrix = get_pairwise_transformation_torch(
#         ego_poses, max_cav=T)

#     # typy2 from origin basedataset
#     # ego_poses_np = ego_poses.cpu().numpy()
#     # spatial_correction_matrix = np.zeros((B,T,4,4))
#     # for i in range(B):
#     #     for j in range(T-1):
#     #         spatial_correction_matrix[i,j] = x1_to_x2(ego_poses_np[i,j], ego_poses_np[i,-1])
#     #     spatial_correction_matrix[i,-1] = np.eye(4)
#     # spatial_correction_matrix = torch.from_numpy(spatial_correction_matrix).to(ego_poses)

#     # project 1 from sttf
#     # features = features.permute(0, 1, 3, 4, 2)         ## features b T h w c
#     # aligned_features = self.fusion_net.encoder.sttf2(features, None, pairwise_t_matrix[:,-1,:,:,:])# 取第三维会波动爆0，取第二维度会跑着跑着除0报错
#     # aligned_features = aligned_features.permute(0, 1, 4, 2, 3)

#     # project 2 from where2coom
#     # import ipdb;ipdb.set_trace()
#     pairwise_t_matrix = pairwise_t_matrix[:, :, :, [
#         0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
#     pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
#     pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
#     _d = self.fusion_net.encoder.downsample_rate * \
#         self.fusion_net.encoder.discrete_ratio
#     pairwise_t_matrix[..., 0,
#                       2] = pairwise_t_matrix[..., 0, 2] / (_d * W) * 2
#     pairwise_t_matrix[..., 1,
#                       2] = pairwise_t_matrix[..., 1, 2] / (_d * H) * 2
#     x_fuse = []
#     for b in range(B):
#         #     # number of valid agent
#         #     # (N,N,4,4)
#         #     # t_matrix[i, j]-> from i to j
#         t_matrix = pairwise_t_matrix[b][:T, :T, :, :]
#         node_features = features[b]
#         neighbor_feature = warp_affine_simple(
#             node_features, t_matrix[target_pos, :, :, :], (H, W))
#         x_fuse.append(neighbor_feature)
#     aligned_features = torch.stack(x_fuse)
#     # aligned_features = torch.cat((aligned_features[:,:-1:,...],features[:,-1:,...]),dim=1)

#     return aligned_features
