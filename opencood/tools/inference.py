import argparse
import os
import time

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
from tqdm import tqdm
from tensorboardX import SummaryWriter

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--calTime', action='store_true',
                        help='only calculate time')
    parser.add_argument('--eval_epoch', type=int, default=-1,
                        help='eval epoch')
    parser.add_argument('--metric',  type=str, default="old",
                        help='eval epoch')
    parser.add_argument('--thresh',  type=float, default="0.01",
                        help='thresh')
    opt = parser.parse_args()
    return opt


def inference(model, opencood_dataset, data_loader, device, fusion_method="intermediate", \
               model_dir="", save_npy=False, writer=None, epoch=0):
    model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create the dictionary for evaluation
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score':[]},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score':[]}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score':[]},
                         0.7: {'tp': [], 'fp': [], 'gt': 0, 'score':[]}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score':[]},
                          0.7: {'tp': [], 'fp': [], 'gt': 0, 'score':[]}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score':[]},
                        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score':[]}}
    for i, batch_data in enumerate(tqdm(data_loader)):
        # print(i)
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            if fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
            elif fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_intermediate_fusion(batch_data,
                                                                 model,
                                                                 opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            # overall calculating
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            # short range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,
                                       left_range=0,
                                       right_range=30)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.7,
                                       left_range=0,
                                       right_range=30)

            # middle range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,
                                       left_range=30,
                                       right_range=50)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.7,
                                       left_range=30,
                                       right_range=50)

            # right range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,
                                       left_range=50,
                                       right_range=100)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.7,
                                       left_range=50,
                                       right_range=100)

            if save_npy:
                npy_save_path = os.path.join(model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)
    
    # range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat,
    #                               model_dir)
    # if writer is not None:
    #     writer.add_scalar('ALL_AP50', ap_50, epoch)
    #     writer.add_scalar('ALL_AP70', ap_70, epoch)

    # range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat_short,
    #                               model_dir,
    #                               "short")
    # if writer is not None:
    #     writer.add_scalar('SHORT_AP50', ap_50, epoch)
    #     writer.add_scalar('SHORT_AP70', ap_70, epoch)

    # range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat_middle,
    #                               model_dir,
    #                               "middle")
    # if writer is not None:
    #     writer.add_scalar('MIDDLE_AP50', ap_50, epoch)
    #     writer.add_scalar('MIDDLE_AP70', ap_70, epoch)

    # range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat_long,
    #                               model_dir,
    #                               "long")
    # if writer is not None:
    #     writer.add_scalar('LONG_AP50', ap_50, epoch)
    #     writer.add_scalar('LONG_AP70', ap_70, epoch)

    ##old_metric

    range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat,
                                  model_dir, metric='old')
    if writer is not None:
        writer.add_scalar('OLD_ALL_AP50', ap_50, epoch)
        writer.add_scalar('OLD_ALL_AP70', ap_70, epoch)

    range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat_short,
                                  model_dir,
                                  "short",
                                  metric='old')
    if writer is not None:
        writer.add_scalar('OLD_SHORT_AP50', ap_50, epoch)
        writer.add_scalar('OLD_SHORT_AP70', ap_70, epoch)

    range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat_middle,
                                  model_dir,
                                  "middle",
                                  metric='old')
    if writer is not None:
        writer.add_scalar('OLD_MIDDLE_AP50', ap_50, epoch)
        writer.add_scalar('OLD_MIDDLE_AP70', ap_70, epoch)

    range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat_long,
                                  model_dir,
                                  "long",
                                  metric='old')
    if writer is not None:
        writer.add_scalar('OLD_LONG_AP50', ap_50, epoch)
        writer.add_scalar('OLD_LONG_AP70', ap_70, epoch)
    

def main(opt):
    '''
    original inference main
    '''       
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
    
    if 'compression_module' in hypes['model']['args']: 
        hypes['model']['args']['compression_module']['thresh']=opt.thresh
        print('opt.thresh: ', opt.thresh)
        
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=opt.show_vis or opt.show_sequence or opt.save_vis, train=False,
                                     isSim=opt.isSim)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    writer = None
    epoch = -1
    saved_path = opt.model_dir
    if opt.eval_epoch > 0:
        # writer = SummaryWriter(saved_path)
        epoch = opt.eval_epoch

    print('Loading Model from checkpoint')
    
    _, model = train_utils.load_saved_model(saved_path, model, initial_epoch=epoch)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score':[]},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score':[]}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score':[]},
                         0.7: {'tp': [], 'fp': [], 'gt': 0, 'score':[]}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score':[]},
                          0.7: {'tp': [], 'fp': [], 'gt': 0, 'score':[]}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0, 'score':[]},
                        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score':[]}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 10
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(500):
            vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            vis_aabbs_pred.append(o3d.geometry.TriangleMesh())
    
    # attn_weights = []
    # hooks = model.temporal_fuse.attn_drop.register_forward_hook(
    #         lambda self, input, output: attn_weights.append(output[0]))
    total_time = []
    for i, batch_data in enumerate(tqdm(data_loader)):
        
        batch_data = train_utils.to_device(batch_data, device)
        
        if opt.calTime:
            from thop import profile, clever_format
            cav_content = batch_data['ego']
            macs, params = profile(model, inputs=(cav_content, True), verbose=False)#
            macs, params = clever_format([macs, params],"%.3f")
            print('macs:',macs)
            print('params:', params)

        # for name, module in model.named_modules():
        #     print(f"{name}: {sum(p.numel() for p in module.parameters() if p.requires_grad)}")
        # breakpoint()

        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            t_ = time.perf_counter()
            if opt.fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
            elif opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_intermediate_fusion(batch_data,
                                                                 model,
                                                                 opencood_dataset,opt.calTime,
                                                                 inference=True)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            t = time.perf_counter() - t_
            print(t)
            total_time.append(t)
            if opt.calTime:
                continue
            # overall calculating
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            # short range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,
                                       left_range=0,
                                       right_range=30)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.7,
                                       left_range=0,
                                       right_range=30)

            # middle range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,
                                       left_range=30,
                                       right_range=50)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.7,
                                       left_range=30,
                                       right_range=50)

            # right range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,
                                       left_range=50,
                                       right_range=100)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.7,
                                       left_range=50,
                                       right_range=100)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = 'vis'
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d_3.png' % i)

                opencood_dataset.visualize_result((pred_box_tensor, pred_score),
                                                  gt_box_tensor,
                                                  batch_data['ego'][-1][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  os.path.join(vis_save_path, '%05d_3.png' % i),
                                                  dataset=opencood_dataset)
                                                #   align_feature=batch_data['ego'][-1]['align_feature'],
                                                #   origin_feature=batch_data['ego'][-1]['origin_feature'],)
                
                ### extra vis for data
                # opencood_dataset.visualize_result(None,
                #                     gt_box_tensor,
                #                     batch_data['ego'][-2][
                #                         'origin_lidar'][0],
                #                     opt.show_vis,
                #                     os.path.join(vis_save_path, '%05d_2.png' % i),
                #                     dataset=opencood_dataset)

                # opencood_dataset.visualize_result(None,
                #                     gt_box_tensor,
                #                     batch_data['ego'][-3][
                #                         'origin_lidar'][0],
                #                     opt.show_vis,
                #                     os.path.join(vis_save_path, '%05d_1.png' % i),
                #                     dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)
    
    if opt.calTime:
        print(sum(total_time) / len(total_time))
        return
    # import ipdb;ipdb.set_trace()
    import numpy as np
    print('communication_rates: ',np.mean(model.communication_rates))
    # print(model.communication_rates)
    range, ap_50, ap_70 = eval_utils.eval_final_results(result_stat,
                                  opt.model_dir, metric=opt.metric)
    if writer is not None:
        writer.add_scalar('ALL_AP50', ap_50, epoch)
        writer.add_scalar('ALL_AP70', ap_70, epoch)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir)
    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir,
                                  "short")
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir,
                                  "middle")
    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir,
                                  "long")

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  metric="old")
    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir,
                                  "short",
                                  metric="old")
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir,
                                  "middle",
                                  metric="old")
    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir,
                                  "long",
                                  metric="old")
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    opt = test_parser()
    # import ipdb;ipdb.set_trace()
    main(opt)
