# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
from telnetlib import IP
from turtle import down
import torch
import torch.nn.functional as F
from torch import nn
import math

from .deformable_transformer import build_deforamble_transformer
import copy
from .position_encoding import PositionEmbeddingSine
from einops.layers.torch import Rearrange, Reduce
from opencood.models.sub_modules.mln import MLN, nerf_positional_encoding, AT


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.ego_id = 0
        self.time_id = -1
        self.prev_len = args['prev_size']

        hidden_dim = args['hidden_dim']
        self.hidden_dim = hidden_dim
        self.pos_emb = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.multi_time = args['multi_time']
        # self.multi_scale = args['multi_scale']
        # self.project_feature = args['proj_feature'] if 'proj_feature' in args else False
        self.dif_offset = args['dif_offset'] if 'dif_offset' in args else False
        self.num_layers = args['enc_layers']

        # if self.project_feature:
        #     self.temporal_align = AT(1)  # velo
        #     self.time_flag = torch.tensor([0, 1])
        #     self.time_delay = torch.tensor([200.0, 0.0])  #20?

        # if self.multi_time:
        #     self.time_transformer = build_deforamble_transformer(
        #         args, num_feature_levels=2, num_encoder_layers=args['enc_layers'])  # recurrent model
        #     self.time_enc = nn.Embedding(2, hidden_dim)
        #     # mlp head
        #     self.time_mlp = nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.LayerNorm(hidden_dim),
        
        #         nn.Linear(hidden_dim, hidden_dim),
        #         Rearrange('b h w c -> b c h w')
        #     )
        #     nn.init.zeros_(self.time_enc.weight)
        
        if self.dif_offset:
            self.query_embed = AT(1)  # velos time_delay and infra, add ego flag?
        
        if self.multi_time:
            max_agents = args['max_cav'] + 1
        else: 
            max_agents = args['max_cav']
        self.transformer = build_deforamble_transformer(
            args, num_feature_levels=max_agents, num_encoder_layers=args['enc_layers'], dif_offset=False)
        
        # self.time_enc = nn.Embedding(max_agents, hidden_dim)
        
        self.mlp_head = nn.Sequential(
            Reduce('b l h w c-> b h w c', 'mean'),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            Rearrange('b h w c -> b c h w')
        )

    def forward(self, src, mask=None, prior_encoding=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # feats [[B L Cx Hx Wx]]  
        # masks B L
        # prior_encoding B L 3 velocity, time_delay, infra
        B, L, C, H, W = src.size()
        if mask is None:
            mask = torch.ones(B, L).bool().to(src.device)
        
        if self.dif_offset:
            src = src + self.query_embed(src, prior_encoding)
            
        src = src.transpose(0, 1).contiguous()  # B, L, C, H, W ->  L B C H W
        mask = mask.transpose(0, 1).contiguous()  # B L-> L B
        mask = mask.reshape(L, B, 1, 1).repeat(1, 1, H, W)  # L B -> L B H W
        pos = self.pos_emb(mask.flatten(0, 1)).reshape(L, B, C, H, W)
        mask = ~mask.bool()  # True for padding tokens

        memory, (spatial_shapes, level_start_index) = self.transformer(src, mask, pos, prior_encoding=prior_encoding)
        out = self.mlp_head(memory.reshape(B, L, H, W, C))

        return out

    # def temporal_fuse_single(self, feats, prior_encoding=None):
    #     # feats B T C H W
    #     B, T, C, H, W = feats.size()
    #     assert T == 2
        
    #     if self.project_feature:
    #         # velos = prior_encoding[...,1]  # B [T0,T1]  C
    #         time_delay = self.time_delay[None].repeat(B, 1).to(feats.device)
    #         time_flag = self.time_flag[None].repeat(B, 1).to(feats.device)  # T -> B T
    #         velos = torch.zeros_like(time_delay)

    #         prior_encoding = torch.stack((velos, time_delay, time_flag), dim=2).view(
    #             B, T, 3)
    #         feats = feats + self.temporal_align(feats, prior_encoding, temporal=True)
        
    #     masks = torch.zeros(T, B, H, W).bool().to(feats.device)
    #     pos = self.pos_emb(masks.flatten(0, 1)).reshape(T, B, C, H, W)
    #     feats = feats.transpose(0, 1).contiguous()  # T B C H W
        
    #     if self.project_feature is False:
    #         pos = pos + self.time_enc.weight.reshape(T, 1, C, 1, 1).repeat(1, B, 1, H, W)
        
    #     memory, _ = self.time_transformer(feats, masks, pos)
    #     out = self.time_mlp(memory.reshape(B, T, H, W, C)[:, self.time_id, ...])
    #     return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
