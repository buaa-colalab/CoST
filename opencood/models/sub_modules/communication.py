from telnetlib import IP
from turtle import update
from cv2 import mean
import torch
import torch.nn as nn
import numpy as np
import random
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from opencood.models.sub_modules.deformable_detr import DeformableDETR


class CompressFuse(nn.Module):  # where2comm
    def __init__(self, args, max_cav=5):
        super(CompressFuse, self).__init__()
        L = max_cav
        self.ego_id = 0
        

        self.smooth = False
        if self.smooth:
            kernel_size = 5
            c_sigma = 1
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

        self.rescale_rate = args['rescale_rate'] if 'rescale_rate' in args else 10
        self.thresh = args['thresh'] if 'thresh' in args else 0.01
        print('thresh:', self.thresh)
        self.reconstruct = args['reconstruct'] if 'reconstruct' in args else False

        if self.reconstruct:  # add dropout?
            input_dim = args['mlp_dim']
            if self.reconstruct == 'deformable':
                self.renew_net = DeformableDETR(args['deformable_transformer'])
            elif self.reconstruct == 'conv':
                self.conv_fuse = nn.Sequential(
                    Rearrange('b l c h w -> (b l) c h w', l=L),
                    nn.Conv2d(2 * input_dim, input_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(input_dim),
                    nn.GELU(),
                    Rearrange('(b l) c h w -> b l c h w', l=L),
                )
            else:
                raise NotImplementedError

            hidden_dim = input_dim // 4
            self.use_decoder = 'use_decoder' in args
            if self.use_decoder is False:
                self.decoder = nn.Identity()
            else:
                self.decoder = nn.Sequential(
                    Rearrange('b l c h w -> (b l) c h w', l=L),
                    nn.Conv2d(input_dim, hidden_dim,
                                kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim,
                                kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, input_dim,
                                kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(input_dim),
                    nn.ReLU(),
                    Rearrange('(b l) c h w -> b l c h w', l=L),
                )

    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center: k_size -
                            center, 0 - center: k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) +
                                                   np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, feat, confidence_map, prev_feat=None, pre_map=None, reset=False):
        # feat is current backbone feature only used in vehicle car  B, L, C, H, W
        # prev_feat is previous aligned backbone feature  B, L, C, H, W
        # confidence_map B L H W
        B, L, _, H, W = confidence_map.shape

        if self.smooth:
            confidence_map = confidence_map.reshape((B * L, 1, H, W))
            communication_maps = self.gaussian_filter(confidence_map)
            communication_maps = communication_maps.reshape((B, L, 1, H, W))

            pre_map = pre_map.reshape((B * L, 1, H, W))
            pre_map = self.gaussian_filter(pre_map)
            pre_map = pre_map.reshape((B, L, 1, H, W))
        else:
            communication_maps = confidence_map
            pre_map = pre_map

        ones_mask = torch.ones_like(communication_maps).to(
            communication_maps.device)
        zeros_mask = torch.zeros_like(
            communication_maps).to(communication_maps.device)
            
        if reset: ## to prevent 
            dynamic_communication_maps = torch.where(
                communication_maps > self.thresh, ones_mask, zeros_mask)
        else:
            dynamic_map = torch.abs(communication_maps - pre_map)  # [0,1]
            dynamic_communication_maps = communication_maps * (1 / (self.rescale_rate + 1) + dynamic_map * self.rescale_rate)
            
            dynamic_communication_maps = torch.where(
                dynamic_communication_maps > self.thresh, ones_mask, zeros_mask)

        dynamic_communication_maps[:, self.ego_id, ...] = 1

        cur_feat = feat * dynamic_communication_maps
        if self.reconstruct:
            if self.use_decoder:
                cur_feat = cur_feat + self.decoder(cur_feat)
            if self.reconstruct == 'conv' and not reset:
                cur_feat = cur_feat + \
                    self.conv_fuse(torch.cat((cur_feat, prev_feat), dim=-3))
            elif self.reconstruct == 'deformable' and not reset:
                cur_feat = cur_feat + self.renew_feat(cur_feat, prev_feat)

        communication_rate = (dynamic_communication_maps[:, self.ego_id+1:,].sum()/(H*W)/B).item()
        # self.all_rates.append(communication_rate.item())
        # print(np.mean(self.all_rates))

        return cur_feat, communication_rate

    def renew_feat(self, sparse_feat, prev_feat):
        B, L, C, H, W = sparse_feat.shape
        fuse_feat = torch.stack((prev_feat, sparse_feat),
                                dim=2).reshape(B * L, 2, C, H, W)
        fuse_feat = self.renew_net(fuse_feat)
        fuse_feat = fuse_feat.reshape(B, L, C, H, W)

        return fuse_feat

# class CompressFuse(nn.Module):
#     def __init__(self, args, max_cav=5):
#         super(CompressFuse, self).__init__()
#         input_dim = args['mlp_dim']
#         dropout = args['dropout']
#         L = max_cav
#         if "compress_rate" in args:
#             self.compress_rate = args['compress_rate']
#         else:
#             self.compress_rate = 0.5
#         self.thresh = 0.01
#         self.sim_fuse = nn.Sequential(
#             Rearrange('b l c h w -> (b l) h w c', l=L),
#             nn.Linear(2 * input_dim, input_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.LayerNorm(input_dim),
#             Rearrange('(b l) h w c -> b l c h w', l=L),
#         )

#         # self.conv_fuse = nn.Sequential(
#         #     nn.Conv2d(2 * input_dim, input_dim, kernel_size=1),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         #     nn.BatchNorm2d(input_dim),
#         # )

#         # self.conv_fuse = nn.Sequential(
#         #     Rearrange('b l c h w -> (b l) c h w', l=L),
#         #     nn.Conv2d(2 * input_dim, input_dim, kernel_size=3, stride=1, padding=1),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         #     nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
#         #     Rearrange('(b l) c h w -> b l c h w', l=L),
#         # )

#         self.channel_compress = args['channel_compress'] if "channel_compress" in args else 0

#         if self.channel_compress > 0:
#             compress_raito = self.channel_compress
#             self.encoder = nn.Sequential(
#                 Rearrange('b l c h w -> (b l) c h w', l=L),
#                 nn.Conv2d(input_dim, input_dim//compress_raito, kernel_size=3,
#                           stride=1, padding=1),
#                 nn.BatchNorm2d(input_dim//compress_raito, eps=1e-3, momentum=0.01),
#                 nn.ReLU()
#             )
#             self.decoder = nn.Sequential(
#                 nn.Conv2d(input_dim//compress_raito, input_dim, kernel_size=3,
#                           stride=1, padding=1),
#                 nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
#                 nn.ReLU(),
#                 nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
#                 nn.BatchNorm2d(input_dim, eps=1e-3,
#                                momentum=0.01),
#                 nn.ReLU(),
#                 Rearrange('(b l) c h w -> b l c h w', l=L),
#             )
#         print("compress_rate", self.compress_rate)
#         print("channel_compress", self.channel_compress)
#         self.attn_drop = nn.Dropout(dropout)

#     def forward(self, prev_feat, feat, confidence_map, pre_map=None,):
#         # feat is current backbone feature only used in vehicle car  B, L, C, H, W
#         # prev_feat is previous aligned backbone feature  B, L, C, H, W
#         # confidence_map B L H W
#         B, L, H, W = confidence_map.shape
#         if len(feat.shape) == 4:
#             feat = feat.reshape(B, L, -1, H, W)
#         if len(prev_feat.shape) == 4:
#             prev_feat = prev_feat.reshape(B, L, -1, H, W)

#         # attn = self.attn_drop(communication_maps)
#         if self.training:
#             # Official training proxy objective
#             K = int(H * W * random.uniform(0, 1))
#         else:
#             K = min(H * W - 1, int(H * W * self.compress_rate))

#         if pre_map is not None:
#             # dynamic_map = torch.abs(confidence_map - pre_map)
#             dynamic_map = confidence_map - pre_map
#             dynamic_map[dynamic_map < 0] = 0  # new target only show in this frame
#             fuse_map = confidence_map + dynamic_map
#         else:
#             fuse_map = confidence_map
#         fuse_map = fuse_map.reshape(B, L, H * W)
#         _, indices = torch.topk(fuse_map, k=K, sorted=False)

#         communication_mask = torch.zeros_like(fuse_map).to(fuse_map.device)
#         ones_fill = torch.ones((B, L, K), dtype=fuse_map.dtype, device=fuse_map.device)
#         communication_mask = torch.scatter(communication_mask, -1, indices, ones_fill).reshape(B, L, 1, H, W)

#         translate_map = communication_mask.squeeze(2) * confidence_map
#         # if not self.training:
#         #     communication_mask[:, 0, ...] = 1  # the ego car do not compress
#         communication_mask[:, 0, ...] = 1  # the ego car do not compress

#         if self.channel_compress > 0:  # compress the channel
#             cur_feat = self.encoder(feat) * communication_mask.reshape(B*L, 1, H, W)
#             cur_feat = self.decoder(cur_feat)
#         else:
#             # cur_feat = feat * communication_mask + prev_feat * (1 - communication_mask)  # fill a new feautre
#             cur_feat = feat * communication_mask
#             # cur_feat = torch.max(cur_feat, prev_feat) ## simple max fusion
#         # cur_feat = cur_feat + prev_feat ## simple add fusion
#         # cur_feat = cur_feat + prev_feat * (1 - communication_mask)

#         cur_feat = torch.cat((cur_feat, prev_feat), dim=-3)
#         # cur_feat = self.conv_fuse(cur_feat)
#         cur_feat = self.sim_fuse(cur_feat)

#         return cur_feat, translate_map
