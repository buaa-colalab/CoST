# -*- coding: utf-8 -*-
# Author: Quanhao Li <quanhaoli2022@163.com> Yifan Lu <yifan_lu@sjtu.edu.cn>, 
# Author: Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for intermediate fusion (DAIR-V2X)
"""
import json
import os
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

import opencood.data_utils.post_processor as post_processor
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, muilt_coord, tfm_to_pose
import bisect

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file, novatel_to_world_json_file):
    matrix = np.empty([4, 4])
    rotationA2B = lidar_to_novatel_json_file["transform"]["rotation"]
    translationA2B = lidar_to_novatel_json_file["transform"]["translation"]
    rotationB2C = novatel_to_world_json_file["rotation"]
    translationB2C = novatel_to_world_json_file["translation"]
    rotation, translation = muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)
    matrix[0:3, 0:3] = rotation
    matrix[:, 3][0:3] = np.array(translation)[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1

    return matrix


def inf_side_rot_and_trans_to_trasnformation_matrix(json_file, system_error_offset):
    matrix = np.empty([4,4])
    matrix[0:3, 0:3] = json_file["rotation"]
    translation = np.array(json_file["translation"])
    translation[0][0] = translation[0][0] + system_error_offset["delta_x"]
    translation[1][0] = translation[1][0] + system_error_offset["delta_y"]  # translation shape (3,1)
    matrix[:, 3][0:3] = translation[:, 0]
    matrix[3, 0:3] = 0
    matrix[3, 3] = 1

    return matrix


class IntermediateFusionDatasetDAIR(Dataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True, isSim=False):
        self.params = params
        self.visualize = visualize
        self.train = train

        #[TODO] many params not used here
        if 'wild_setting' in params:
            ##tzh
            interval_size = 1
            if 'interval_size' in params['wild_setting']:
                interval_size = params['wild_setting']['interval_size']
            self.prev_len = [interval_size * i for i in range(params['wild_setting']['prev_size']-1, -1, -1)]
            self.seed = params['wild_setting']['seed']
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed']\
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0 # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0 # Mb
            self.transmission_speed = 27 # Mbps
            self.backbone_delay = 0 # ms

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']
        assert self.max_cav >= 2

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        # if 'proj_first' in params['fusion']['args'] and \
        #         not params['fusion']['args']['proj_first']:
        #     self.proj_first = False

        self.augment_config = params['data_augment']
        self.data_augmentor = DataAugmentor(params['data_augment'], train, intermediate=True)

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        self.root_dir = params['dair_data_dir']
        # import ipdb;ipdb.set_trace()
        split = 'train' if train or 'pseudo_label_generation' in params else 'val'
        self.split_info = load_json(os.path.join(self.root_dir, f'{split}.json'))
        co_datainfo = load_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_pointcloud_path'].split("/")[-1].replace(".pcd", "")
            self.co_data[veh_frame_id] = frame_info

        self.inf_path2info = build_path_to_info(
            "infrastructure-side",
            load_json(osp.join(self.root_dir, "infrastructure-side/data_info.json")),
            'lidar',
        )
        self.veh_path2info = build_path_to_info(
            "vehicle-side",
            load_json(osp.join(self.root_dir, "vehicle-side/data_info.json")),
            'lidar',
        )
        self.reinitialize()
        print('use_prev frames: ', self.prev_len)

    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.veh2batchid = {}
        self.batchid2index = {}
        self.index2batchid = {}

        for veh_frame_path in self.veh_path2info:
            # import ipdb;ipdb.set_trace()
            veh_frame_id = veh_frame_path.split("/")[-1].replace(".pcd", "")
            veh_frame = self.veh_path2info[veh_frame_path]
            if veh_frame_id not in self.split_info:
                continue
            self.veh2batchid[veh_frame_id] = veh_frame["batch_id"]
        last = -1
        for i in range(len(self.split_info)):
            veh_frame_id = self.split_info[i]
            # only used once
            assert int(veh_frame_id) > last
            last = int(veh_frame_id)
            batchid = self.veh2batchid[veh_frame_id]
            self.index2batchid[i] = batchid
            if not batchid in self.batchid2index:
                self.batchid2index[batchid] = [i]
            else:
                assert i == self.batchid2index[batchid][-1]+1
                self.batchid2index[batchid].append(i)
                
        # import ipdb;ipdb.set_trace()
        return
            
    def generate_augment(self):
        flip = [None, None, None]
        noise_rotation = None
        noise_scale = None
        
        if self.augment_config is None:
            return flip, noise_rotation, noise_scale

        for aug_ele in self.augment_config:
            # for intermediate fusion only
            if 'random_world_rotation' in aug_ele['NAME']:
                rot_range = \
                    aug_ele['WORLD_ROT_ANGLE']
                if not isinstance(rot_range, list):
                    rot_range = [-rot_range, rot_range]
                noise_rotation = np.random.uniform(rot_range[0])

            if 'random_world_flip' in aug_ele['NAME']:
                for i, cur_axis in enumerate(aug_ele['ALONG_AXIS_LIST']):
                    enable = np.random.choice([False, True], replace=False,
                                              p=[0.5, 0.5])
                    flip[i] = enable

            if 'random_world_scaling' in aug_ele['NAME']:
                scale_range = \
                    aug_ele['WORLD_SCALE_RANGE']
                noise_scale = \
                    np.random.uniform(scale_range[0], scale_range[1])

        return flip, noise_rotation, noise_scale

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()
        data[0] = OrderedDict() # veh-side
        data[0]['ego'] = True
        data[1] = OrderedDict() # inf-side
        data[1]['ego'] = False
 
        data[0]['params'] = OrderedDict()

        data[0]['params']['vehicles_coop'] = load_json(os.path.join(self.root_dir, frame_info['cooperative_label_path']))

        lidar_to_novatel_json_file = load_json(os.path.join(self.root_dir,f'vehicle-side/calib/lidar_to_novatel/{veh_frame_id}.json'))
        novatel_to_world_json_file = load_json(os.path.join(self.root_dir,f'vehicle-side/calib/novatel_to_world/{veh_frame_id}.json'))
        transformation_matrix_v = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel_json_file,
                                                                                novatel_to_world_json_file)
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix_v)

        ######################## Single View GT ########################
        vehicle_side_path = os.path.join(self.root_dir, f'vehicle-side/label/lidar/{veh_frame_id}.json')

        data[0]['params']['vehicles'] = load_json(vehicle_side_path)

        ######################## Single View GT ########################

        data[0]['lidar_np'] = pcd_utils.read_pcd(os.path.join(self.root_dir, frame_info["vehicle_pointcloud_path"]))

        data[1]['params'] = OrderedDict()
        inf_frame_id = frame_info['infrastructure_pointcloud_path'].split("/")[-1].replace(".pcd", "")
        data[1]['params']['vehicles_coop'] = [] # we only load cooperative once in veh-side
        virtuallidar_to_world_json_file = load_json(os.path.join(self.root_dir,
                                                                 f'infrastructure-side/calib/virtuallidar_to_world/{inf_frame_id}.json'))
        transformation_matrix_i = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world_json_file, system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix_i)

        ######################## Single View GT ########################
        infra_side_path = os.path.join(self.root_dir, f'infrastructure-side/label/virtuallidar/{inf_frame_id}.json')

        data[1]['params']['vehicles'] = load_json(infra_side_path)

        ######################## Single View GT ########################

        data[1]['lidar_np'] = pcd_utils.read_pcd(os.path.join(self.root_dir, frame_info["infrastructure_pointcloud_path"]))
        # import ipdb;ipdb.set_trace()
        return data

    def __len__(self):
        return len(self.split_info)
    
    def get_item_single_car(self, selected_cav_base, ego_pose, ego_pose_clean):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate with noise.
        ego_pose_clean : list, length 6
            The ego vehicle lidar pose under world coordinate without noise. Only used for gt box generation.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose) # T_ego_cav
        transformation_matrix_clean = \
            x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)

        # retrieve objects under ego coordinates
        # this is used to generate accurate GT bounding box.

        object_bbx_center_coop, object_bbx_mask_coop, object_ids_coop = \
            self.post_processor.generate_object_center_dairv2x([selected_cav_base], ego_pose_clean)

        object_bbx_center, object_bbx_mask, object_ids = self.post_processor.generate_object_center_dairv2x_late_fusion(
            [selected_cav_base])

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)

        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)

        # data augmentation
        lidar_np, object_bbx_center_coop, object_bbx_mask_coop = \
            self.augment(lidar_np, object_bbx_center_coop, object_bbx_mask_coop,
                         selected_cav_base['flip'],
                         selected_cav_base['noise_rotation'],
                         selected_cav_base['noise_scale'])
        # filter out the augmented bbx that is out of range
        object_bbx_center_coop_valid = object_bbx_center_coop[object_bbx_mask_coop == 1]
        object_bbx_center_coop_valid, range_mask = \
            box_utils.mask_boxes_outside_range_numpy(object_bbx_center_coop_valid,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'],
                                                     self.params['postprocess'][
                                                         'order'])
        object_ids_coop = [int(x) for x in list(np.array(object_ids_coop)[range_mask])]
            
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess']['cav_lidar_range'])
        processed_lidar = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update(
            {'object_bbx_center_coop': object_bbx_center_coop_valid,
             # 'object_bbx_center_coop': object_bbx_center_coop[object_bbx_mask_coop == 1],
             'object_ids_coop': object_ids_coop,
             'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'transformation_matrix': transformation_matrix,
             'transformation_matrix_clean': transformation_matrix_clean})

        return selected_cav_processed

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask, flip=None, rotation=None, scale=None):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask,
                    'flip': flip,
                    'noise_rotation': rotation,
                    'noise_scale': scale}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def get_unique_label(self, object_stack, object_id_stack):
        # IoU
        object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])

        if len(object_stack) > 0:
            # exclude all repetitive objects    
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack) if len(object_stack) > 1 else object_stack[0]
            object_stack = object_stack[unique_indices]
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            updated_object_id_stack = [object_id_stack[i] for i in unique_indices]
        else:
            updated_object_id_stack = object_id_stack

        return object_bbx_center, mask, updated_object_id_stack

    @staticmethod
    def add_noise_data_dict(data_dict, noise_setting):
        """ Update the base data dict.
            We retrieve lidar_pose and add_noise to it.
            And set a clean pose.
        """
        if noise_setting['loc_err']:
            for cav_id, cav_content in data_dict.items():
                cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'].copy() # 6 dof pose
                xy_noise = np.random.normal(0, noise_setting['xyz_std'], 2)
                yaw_noise = np.random.normal(0, noise_setting['ryp_std'], 1)
                pose_noise = np.array([xy_noise[0], xy_noise[1], 0, 0, yaw_noise[0], 0])
                cav_content['params']['lidar_pose'] += pose_noise
                # cav_content['params']['lidar_pose'][:2] += xy_noise
                # cav_content['params']['lidar_pose'][5:] += yaw_noise  ###!!!!
        else:
            for cav_id, cav_content in data_dict.items():
                cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose

        return data_dict

    def __getitem__(self, idx):

        aug = self.generate_augment() 
        processed_data_dict = [] 
        # import ipdb;ipdb.set_trace()
        #确定不要越界
        batchid = self.index2batchid[idx]
        segment = self.batchid2index[batchid]
        start_idx = segment[0]

        for i in self.prev_len:
            if idx-self.prev_len[0] >= start_idx: 
                processed_data_dict.append(self.__getitem__ori(idx-i, aug)) 
            else:
                processed_data_dict.append(self.__getitem__ori(idx, aug))## to fix?
            processed_data_dict[-1]['ego']['prev_exist'] = idx-i > start_idx
            
        return processed_data_dict

    def __getitem__ori(self, idx, aug=None):
        # import ipdb;ipdb.set_trace()
        base_data_dict = self.retrieve_base_data(idx)

        if 'wild_setting' in self.params:
            base_data_dict = self.add_noise_data_dict(base_data_dict, self.params['wild_setting'])
        else:
            base_data_dict = self.add_noise_data_dict(base_data_dict, {'loc_err': False})

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {'frame_id': self.split_info[idx]}

        if aug is not None:
            flip, noise_rotation, noise_scale = aug
        else:
            # augmentation related
            flip, noise_rotation, noise_scale = self.generate_augment()
            # print(noise_rotation)


        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                break
            
        assert cav_id == 0, "0 must be ego"
        assert ego_id == 0
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = self.get_pairwise_transformation(base_data_dict, self.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []
        object_stack_single_v = []
        object_id_stack_single_v = []
        object_stack_single_i = []
        object_id_stack_single_i = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        cav_id_list = []

        # For pseudo label
        object_stack_low_thresh = []
        object_id_stack_low_thresh = []

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
            cav_id_list.append(cav_id)

        for cav_id in cav_id_list:
            selected_cav_base = base_data_dict[cav_id]

            # augment related
            selected_cav_base['flip'] = flip
            selected_cav_base['noise_rotation'] = noise_rotation
            selected_cav_base['noise_scale'] = noise_scale
            
            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose, 
                ego_lidar_pose_clean)
            # import ipdb;ipdb.set_trace()
            object_stack.append(selected_cav_processed['object_bbx_center_coop'])
            object_id_stack += selected_cav_processed['object_ids_coop']

            ######################## Single View GT ########################
            if cav_id == 0:
                object_stack_single_v.append(selected_cav_processed['object_bbx_center'])
                object_id_stack_single_v += selected_cav_processed['object_ids']
            else:
                object_stack_single_i.append(selected_cav_processed['object_bbx_center'])
                object_id_stack_single_i += selected_cav_processed['object_ids']
            ######################## Single View GT ########################

            processed_features.append(selected_cav_processed['processed_features'])

            velocity.append(0)
            time_delay.append(0)
            # import ipdb;ipdb.set_trace()
            # velocity.append(selected_cav_processed['velocity'])
            # time_delay.append(float(selected_cav_base['time_delay']))

            spatial_correction_matrix.append(np.eye(4))
            infra.append(1 if int(cav_id) == 1 else 0)

            # if self.visualize:
            #     # raise NotImplementError('Visualization not implemented yet')
            #     projected_lidar = selected_cav_processed['projected_lidar']
            #     projected_lidar = \
            #         np.r_[projected_lidar.T,
            #               [np.ones(projected_lidar.shape[0])*i]].T

            #     projected_lidar_stack.append(projected_lidar)
            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

        # import ipdb;ipdb.set_trace()
        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]

        object_bbx_center, mask, object_id_stack = self.get_unique_label(object_stack, object_id_stack)
        
        ######################## Single View GT ########################
        object_bbx_center_single_v, mask_single_v, object_id_stack_single_v = self.get_unique_label(object_stack_single_v, object_id_stack_single_v)
        object_bbx_center_single_i, mask_single_i, object_id_stack_single_i = self.get_unique_label(object_stack_single_i, object_id_stack_single_i)
        ######################## Single View GT ########################

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)
        
        label_dict_single_v = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center_single_v,
                anchors=anchor_box,
                mask=mask_single_v)
        
        label_dict_single_i = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center_single_i,
                anchors=anchor_box,
                mask=mask_single_i)

        # pad dv, dt, infra to max_cav
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav -
            len(spatial_correction_matrix), 1, 1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                    padding_eye], axis=0)
        #[TODO] merge label_dict_single_v and label_dict_single_i to label_dict_all,
        label_dict_all = []
        label_dict_all.append(label_dict_single_v)
        label_dict_all.append(label_dict_single_i)
        # import ipdb;ipdb.set_trace()
        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': object_id_stack,
             'label_dict': label_dict,
             'object_bbx_center_single_v': object_bbx_center_single_v,
             'object_bbx_mask_single_v': mask_single_v,
             'object_ids_single_v': object_id_stack_single_v,
             'label_dict_single_v': label_dict_single_v,
             'object_bbx_center_single_i': object_bbx_center_single_i,
             'object_bbx_mask_single_i': mask_single_i,
             'object_ids_single_i': object_id_stack_single_i,
             'label_dict_single_i': label_dict_single_i,
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'cav_num': cav_num,
             'velocity': velocity,
             'time_delay': time_delay,
             'infra': infra,
             'pairwise_t_matrix': pairwise_t_matrix,
             'lidar_poses_clean': lidar_poses_clean,
             'lidar_poses': lidar_poses,
             'spatial_correction_matrix': spatial_correction_matrix,
             'ego_lidar_pose': ego_lidar_pose,
             'label_dict_all': label_dict_all,
             })

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})

            processed_data_dict['ego'].update({'origin_lidar_v':
                    projected_lidar_stack[0]})
            processed_data_dict['ego'].update({'origin_lidar_i':
                    projected_lidar_stack[1]})

        return processed_data_dict
    
    def collate_batch_train(self, batch):
        batch = [list(x) for x in zip(*batch)]  ## batch的shape[batch,prev_len], 反转一下
        # print(len(batch), len(batch[0]))
        # import ipdb;ipdb.set_trace()
        return {'ego': [self.collate_batch_train_ori(batch[i])['ego'] for i in range(len(self.prev_len))] }
        
    def collate_batch_train_ori(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        label_dict_list = []
        label_dict_all_list=[]

        ######################## Single View GT ########################
        object_bbx_center_single_v = []
        object_bbx_mask_single_v = []
        object_ids_single_v = []
        label_dict_list_single_v = []

        object_bbx_center_single_i = []
        object_bbx_mask_single_i = []
        object_ids_single_i = []
        label_dict_list_single_i = []
        ######################## Single View GT ########################

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []
        ego_pose_list = []

        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        
        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        spatial_correction_matrix_list = []
        # edo pose for temporal alignment


        prev_exist = []
        if self.visualize:
            origin_lidar = []
            origin_lidar_v = []
            origin_lidar_i = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            label_dict_list.append(ego_dict['label_dict'])
            label_dict_all_list.append(self.post_processor.collate_batch(ego_dict['label_dict_all']))

            ######################## Single View GT ########################
            object_bbx_center_single_v.append(ego_dict['object_bbx_center_single_v'])
            object_bbx_mask_single_v.append(ego_dict['object_bbx_mask_single_v'])
            object_ids_single_v.append(ego_dict['object_ids_single_v'])
            label_dict_list_single_v.append(ego_dict['label_dict_single_v'])

            object_bbx_center_single_i.append(ego_dict['object_bbx_center_single_i'])
            object_bbx_mask_single_i.append(ego_dict['object_bbx_mask_single_i'])
            object_ids_single_i.append(ego_dict['object_ids_single_i'])
            label_dict_list_single_i.append(ego_dict['label_dict_single_i'])
            ######################## Single View GT ########################
            
            lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
            lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

            processed_lidar_list.append(ego_dict['processed_lidar']) # different cav_num, ego_dict['processed_lidar'] is list.
            record_len.append(ego_dict['cav_num'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])
            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(ego_dict['spatial_correction_matrix'])
            ego_pose_list.append(ego_dict['ego_lidar_pose'])

            if 'prev_exist' in ego_dict.keys():
                prev_exist.append(ego_dict['prev_exist'])
            else:
                prev_exist.append(False)

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
                origin_lidar_v.append(ego_dict['origin_lidar_v'])
                origin_lidar_i.append(ego_dict['origin_lidar_i'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        ######################## Single View GT ########################
        object_bbx_center_single_v = torch.from_numpy(np.array(object_bbx_center_single_v))
        object_bbx_mask_single_v = torch.from_numpy(np.array(object_bbx_mask_single_v))

        object_bbx_center_single_i = torch.from_numpy(np.array(object_bbx_center_single_i))
        object_bbx_mask_single_i = torch.from_numpy(np.array(object_bbx_mask_single_i))
        ######################## Single View GT ########################

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)

        # [sum(record_len), C, H, W]
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        label_torch_dict_single_v = \
            self.post_processor.collate_batch(label_dict_list_single_v)
        label_torch_dict_single_i = \
            self.post_processor.collate_batch(label_dict_list_single_i)
        label_torch_dict_all = \
            self.post_processor.collate_batch(label_dict_all_list, True)
        # (B, max_cav)
        prev_exist =  torch.from_numpy(np.array(prev_exist, dtype=bool))
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix = torch.from_numpy(np.array(spatial_correction_matrix_list))

        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()
        #ego_pose_list
        ego_pose_list = torch.from_numpy(np.array(ego_pose_list))
        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len

        ######################## Single View GT ########################
        label_torch_dict_single_v['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict_single_v['record_len'] = record_len

        label_torch_dict_single_i['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict_single_i['record_len'] = record_len
        ######################## Single View GT ########################

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'object_ids': object_ids[0],
                                   'label_dict': label_torch_dict,
                                   'object_bbx_center_single_v': object_bbx_center_single_v,
                                   'object_bbx_mask_single_v': object_bbx_mask_single_v,
                                   'object_ids_single_v': object_ids_single_v[0],
                                   'label_dict_single_v': label_torch_dict_single_v,
                                   'object_bbx_center_single_i': object_bbx_center_single_i,
                                   'object_bbx_mask_single_i': object_bbx_mask_single_i,
                                   'object_ids_single_i': object_ids_single_i[0],
                                   'label_dict_single_i': label_torch_dict_single_i,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'pairwise_t_matrix': pairwise_t_matrix,
                                   'lidar_pose_clean': lidar_pose_clean,
                                   'lidar_pose': lidar_pose,
                                   'spatial_correction_matrix': spatial_correction_matrix,
                                   'prior_encoding': prior_encoding,
                                   'ego_pose': ego_pose_list,
                                   'prev_exist':prev_exist, 
                                   'label_dict_all':label_torch_dict_all})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

            origin_lidar_v = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar_v))
            origin_lidar_v = torch.from_numpy(origin_lidar_v)
            output_dict['ego'].update({'origin_lidar_v': origin_lidar_v})
        
            origin_lidar_i = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar_i))
            origin_lidar_i = torch.from_numpy(origin_lidar_i)
            output_dict['ego'].update({'origin_lidar_i': origin_lidar_i})

        return output_dict

    def collate_batch_test(self, batch):
        batch = [list(x) for x in zip(*batch)]  ## batch的shape[batch,prev_len], 反转一下
        # print(len(batch), len(batch[0]))
        # import ipdb;ipdb.set_trace()
        return {'ego': [self.collate_batch_test_ori(batch[i])['ego'] for i in range(len(self.prev_len))]}
    
    def collate_batch_test_ori(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train_ori(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(batch[0]['ego']['anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = \
            torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch,
                                   'transformation_matrix_clean':
                                       transformation_matrix_clean_torch,
                                   'frame_id': batch[0]['ego']['frame_id']})

        return output_dict
    
    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1)) # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['params']['lidar_pose']
                t_list.append(x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
    
    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature) # merged_feature_dict['coords'] = [f1,f2,f3,f4]
        return merged_feature_dict
    
    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None,
                         align_feature=None,
                         origin_feature=None,
                         ):
        # import ipdb;ipdb.set_trace()
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset,
                                      align_feature=align_feature,
                                      origin_feature=origin_feature)
def build_path_to_info(prefix, data, sensortype="lidar"):
    path2info = {}
    if sensortype == "lidar":
        for elem in data:
            if elem["pointcloud_path"] == "":
                continue
            path = osp.join(prefix, elem["pointcloud_path"])
            path2info[path] = elem
    elif sensortype == "camera":
        for elem in data:
            if elem["image_path"] == "":
                continue
            path = osp.join(prefix, elem["image_path"])
            path2info[path] = elem
    return path2info
