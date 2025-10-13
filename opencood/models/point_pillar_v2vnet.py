import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.v2v_fuse import V2VNetFusion
from opencood.models.sub_modules.split_attn import SplitAttn2
from opencood.utils.transformation_utils import align_features

class PointPillarV2VNet(nn.Module):
    def __init__(self, args):
        super(PointPillarV2VNet, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2VNetFusion(args['v2vfusion'])
        self.temporal_type = args['temporal_type']
        assert self.temporal_type == 1 or self.temporal_type == 0

        if self.temporal_type == 1:
            self.temporal_fuse = SplitAttn2(256)
            
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dicts, return_loss=True):
        # import ipdb;ipdb.set_trace()
        if self.temporal_type == 0:
            return self.forward_ori(data_dicts[-1])
        elif self.temporal_type == 1:
            return self.forward_split(data_dicts, return_loss)

    def unpad_prior_encoding(self, x, record_len):
        # remove padded zeros to form tensor with shape (N, 3)
        # x: (B, L, 3); record_len: (B)
        B = x.shape[0]
        out = []
        for i in range(B):
            # (valid_len, 3)
            out.append(x[i, :record_len[i], :])
        out = torch.cat(out, dim=0)
        # (N, 3)
        return out

    def forward_split(self, data_dicts, train):
        self.eval()
        with torch.no_grad():
            prev_bev = [self.forward_ori(data_dict, only_bev=True)
                        for data_dict in data_dicts[:-1]]
        if train:
            self.train()

        fused_feature = self.forward_ori(
            data_dicts[-1], only_bev=True)  # b c h w
        prev_bev.append(fused_feature)  # get the

        prev_bev = torch.stack(prev_bev).transpose(0, 1)
        ego_poses = torch.stack([data_dict['ego_pose']
                                for data_dict in data_dicts]).transpose(0, 1)                       
        prev_bev = align_features(prev_bev, ego_poses).transpose(0, 1)
        # prev_nev2 = self.align(prev_bev, ego_poses).transpose(0,1)
        # bs cav c h w
        fused_feature = self.temporal_fuse(prev_bev)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                    'rm': rm}
        return output_dict
        
    def forward_ori(self, data_dict, only_bev=False,):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        prior_encoding = data_dict['prior_encoding']
        prior_encoding = self.unpad_prior_encoding(prior_encoding, record_len)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        fused_feature = self.fusion_net(spatial_features_2d,
                                        record_len,
                                        pairwise_t_matrix,
                                        prior_encoding)
        if only_bev:
            return fused_feature

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
